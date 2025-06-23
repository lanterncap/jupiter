from abc import ABC, abstractmethod
import numpy as np

class Feature(ABC):
    """
    Base class for composable low-bitwidth features with temporal management.

    Each feature must implement:
        - value(): returns a raw signal value (float)

    Core responsibilities:
        - Map raw value to a discrete bucket (with boundaries defined in `edges`)
        - Map bucket index to a confidence score via `scores[bucket]`
    
    Temporal management:
        - Maintain feature state across time using trigger + decay mechanics
        - Trigger: Occurs when bucket is in `trigger_buckets`, the feature is considered 
        activated, and `last_score` is stored. If unspecified, default to the bucket with the 
        highest score. The use of `trigger_buckets` enables activation in specific score regions only,
        such as mid-range confidence zones in non-monotonic profiles
        - Decay phase: For the next `len(score_decays)` bars, score = last_score * score_decays[bars_since_trigger].
        This includes the trigger bar (at index 0), allowing deferred or partial signal activation via score_decays[0]
        - After decay phase ends, the feature resumes normal bucket-to-score conversion based on current raw 
        values and may retrigger based on the recomputed bucket
        - Potential use cases:
            - Delayed signal fusion: a strong signal would remain relevant over several bars with
            decaying confidence (e.g. [1, 0.75, 0.5, 0.25])
            - Relevance window: use flat decay to represent stable signals over a few bars
            (e.g. [1, 1, 1, 1] for completed candle patterns on a higher timeframe)
            - Cooldown/fadeout: use negative scores after expiry to discourage repeated or failed
            signals (e.g. [1, 0.5, 0, -1])
            
    Feature as an independent signal:
        - Each feature encodes a hypothesis and can be independently testable
        - Design choice: score must be monotonic (even when raw values aren't). Higher score means 
        higher confidence in desired trading outcome. Without this, composite scores would be 
        uninterpretable or misleading
        - Top score does not have to be 1.0; can assign for convenience then fine-tune
        - Raw value need not be monotonic. Extremely high values may signal exhaustion or reversal, 
        leading to a U- or hump-shaped return profile. For example: edges = [2.5, 3.2, 4.5, 6.0], 
        scores = [0, 0.7, 1, 0.5, 0]
        - Examples of techniques to use backtest results to fit score mappings:
            - Score is proportional to bucket pnl (relative to no-signal pnl)
            - Scale down bucket score if too few signals,  by sqrt(num_trade / max_num_trade) across signal buckets
            - Combine buckets, flatten, or boost scores while maintaining monotonicity
            - Use bucket's Sharpe ratio instead of pnl when having more samples (>50 per bucket)
    
    Feature used in a composite signal:
        - A feature's score represents its confidence contribution to a composite signal
        - The optimal score mapping may differ from its standalone calibration due to interactions 
        with other features and market regime dependencies
        - Retuning scores in context is valid, but should preserve the feature's original behavioral logic and 
        guard against overfitting
        - Composite score behavior depends on the monotonicity and alignment of individual feature scores; abrupt 
        or noisy mappings can destabilize signal quality
        - Composite scores are not just aggregations—they encode the joint confidence surface across multiple features
        - If a feature is highly correlated with other features, it may be redundant and should be 
        considered for removal or reweighting
        - Conditional signal emergence: a feature may first be useful only as a gating condition, later 
        show predictive strength in a filtered context, and ultimately become a scored component in the 
        composite (e.g., ADX, RSI)
        - Some features may be useful both as a standalone regime selector (e.g. only trade if ADX > 15) and as a
        composite component (e.g. ADX in 15-60 correlates monotonically with trade results)

    Scoring and binary features:
        - Scoring features: continuous or ordinal features with scores indicating signal strength 
        (e.g., momentum, analyst ratings), often aggregated into composite scores for ensemble models
        - Binary features (also called indicator features, distinct from technical indicators): features 
        yielding a True/False or 1/0 result (e.g., breakout present), used as regime selectors, trade 
        triggers, or model inputs; also called categorical features if producing more than two unordered results
    """

    CONFIG = {} # regime -> config (dict)

    @classmethod
    def build_feature(cls, strategy, **params):
        return cls(strategy, **params)
                        
    def __init__(self, strategy, name=None, edges=None, scores=None, trigger_buckets=None, score_decays=None, **kwargs):
        self.strategy = strategy
        self.name = name or self.__class__.__name__
        self.edges = edges or [0]
        self.scores = scores or [i/len(self.edges) for i in range(len(self.edges)+1)] # default linear scores, e.g. [0, 0.25, 0.5, 0.75, 1]
        
        # temporal management
        self.trigger_buckets = trigger_buckets or {np.argmax(self.scores)} # default to bucket with highest score
        self.score_decays = score_decays or 1
        if isinstance(self.score_decays, int): # build a linear decay profile if not provided
            self.score_decays = [1 - i/self.score_decays for i in range(self.score_decays)]  # e.g. 4 -> [1, 0.75, 0.5, 0.25]
        self.last_score = 0.0
        self.trigger_idx = -len(self.score_decays) # no detection yet
        
        self.logs = {}
        self.enabled = True

        # run once to put log fields in preferred order
        self.log("value", None)
        self.log("bucket", None)
        self.log("score", None)
    
    @abstractmethod
    def value(self):
        """
        Computes the raw signal value for this feature.
        """
        pass

    def reject_value(self):
        """
        Returns a value guaranteed to fall into the lowest bucket, indicating no meaningful signal at this time.
        """
        val = self.edges[0] - 1e-6
        self.log("value", val)
        return val

    def bucket(self):
        """
        Maps value to an integer bucket index.
        """
        val = self.value()
        bucket = next((i for i, edge in enumerate(self.edges) if edge > val), len(self.edges))
        self.log("bucket", bucket)
        return bucket
            
    def score(self):
        """
        Converts bucket index into a normalized [0, 1] score, decaying over time if enabled.
        """
        if len(self.score_decays) <= 1: # no temporal management
            score = self.scores[self.bucket()]
            self.log("score", score)
            return score
        
        # temporal management enabled
        bars_since_trigger = len(self.strategy.data) - self.trigger_idx
        if bars_since_trigger < len(self.score_decays): # in decay phase
            score = self.last_score * self.score_decays[bars_since_trigger]
        else: # decay phase ended
            bucket = self.bucket() # re-compute bucket
            triggered = bucket in self.trigger_buckets
            score = self.scores[bucket] * (self.score_decays[0] if triggered else 1) # e.g. decays[0] = 0 for delayed activation
            self.last_score = self.scores[bucket] if triggered else None
            self.trigger_idx = len(self.strategy.data) if triggered else self.trigger_idx
            bars_since_trigger = 0 if triggered else -1 # -1 to indicate not in decay phase

        self.log("original_score", self.last_score)
        self.log("bars_since_trigger", bars_since_trigger)
        self.log("score", score)
        return score

    def to_dict(self, weight=None):
        if weight is not None:
            self.log("weight", round(weight, 4))
        return self.logs
    
    def log(self, key, value):
        self.logs[f"{self.name}_{key}"] = value

class FeatureEntry:
    """
    Represents a feature and its associated weight inside a CompositeSignal.
    """
    def __init__(self, feature, weight):
        self.feature = feature
        self.weight = weight

class CompositeSignal:
    """
    Composite signal that aggregates multiple low-bitwidth feature scores.

    Features are registered via `add_feature()` with associated weights.
    Computes weighted average score and checks if it exceeds a threshold.

    Signal score intepretation examples:
    -------------------------------------
    - 0.0: no signal
    - 0.2: weak
    - 0.6: average
    - 1.0: strong
    """

    def __init__(self, threshold=0.75):
        self.features = {}  # name -> FeatureEntry
        self.threshold = threshold
        self.logs = {}

    def add_feature(self, feature, weight=1.0):
        if feature.name in self.features:
            print(f"Feature '{feature.name}' already exists.")
            return
        self.features[feature.name] = FeatureEntry(feature, weight)

    def score(self):
        score = 0.0
        total_weight = 0.0
        for entry in self.features.values():
            if entry.feature.enabled:
                score += entry.feature.score() * entry.weight
                total_weight += entry.weight
        score = score / total_weight if total_weight > 0 else 0.0
        self.logs["signal_score"] = score
        return score

    def triggered(self):
        triggered = self.score() >= self.threshold
        self.logs["signal_triggered"] = triggered
        return triggered
    
    def to_dict(self):
        out = {}
        for entry in self.features.values():
            if entry.feature.enabled:
                out.update(entry.feature.to_dict(entry.weight))
        out.update(self.logs)
        return out
    
    def log(self, key, value):
        self.logs[key] = value

    def update_logs(self, dict):
        self.logs.update(dict)
    
    def clear_logs(self):
        self.logs = {}
