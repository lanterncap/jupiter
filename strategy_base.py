"""
Entry logic is hierarchically constructed:
- entry_pre_condition: per-bar baseline conditions such as time filters, inexpensive to evaluate
- entry_context_condition: context or regime classification (e.g., trend, volatility zone); evaluated 
infrequently (e.g., per session or every N bars), with results stored in flags to avoid race conditions
- entry_setup_condition: potential trading conditions without immediate triggers; evaluated when context allows
- entry_signal: immediate microstructural confirmation; evaluated when setup conditions are met
- entry_execution: places orders after all conditions are met, updates flags and status

Exit logic mirrors the above structure

Hypothesis Decomposition Framework for Strategy Design:
- Catalyst: What market event or condition changes participants' perceptions?
- Behavioral Response: How do different groups of participants respond?
- Early Evidence: What observable evidence or indicators appear early?
- Prediction: What is likely to happen next?
- Regime/Assumptions: Under what conditions is this hypothesis valid?

Behavioral archtype families (examples):
- News-driven leading to trend extension
- Liquidity vacuum leading to reversal
- End-of-session ramp leading to directional close
- Overnight gap continuation leading to trend extension
- Overnight gap reversal leading to trend reversal

Example 1: News-driven smart money entry
- Catalyst: news release
- Behavioral response: retail traders react first (noisy); institutions enter more methodically to build real positions
- Early evidence: volume surge + wide body candles + rising ADX
- Prediction: trend continuation 30-90 minutes after news
- Regime/Assumptions: work bests in medium volatility (not chaostic panic); fails in low-volume 
regime. Difficult to trades if news whipsaws (e.g. conflicting headlines)

Example 2: Liquidity vacuum leading to reversal
- Catalyst: sudden price move causes a liquidity gap (e.g., flash crash, large order)
- Behavioral response: market makers widen spreads, causing exaggerated price moves
- Early evidence: extreme 1-bar move in price + low follow-up volume
- Predicted consequence: sharp mean-reversion snapback toward prior levels
- Regime/Assumptions: works best in low liquidity periods (midday, late day). Fails during 
genuine news-driven breaks

Example 3: Open gap exhaustion
- Catalyst: big overnight gap caused by news
- Behavioral response: early players (retail traders) exhaust themselves trying to chase continuation
- Early evidence: open outside prior day range + fast retracement with rising volume
- Predicted consequence: gap fills back toward prior close
- Regime/assumptions: works best in low-volatility, non-trending broader market conditions. Fails 
in very strong directional trending days

------------------------------------------------------------------------------
Naming Convention

1. STRATEGY_CODE
   - Format: "HORIZON_ARCHETYPE_ENTRY_[VARIANT1_VARIANT2...]_[VERSION]"
   - Example: "DAILY_REVERSION_VWAP_V1"
   - Example (extended): "DAILY_REVERSION_VWAP_LOWVOL_EOD_V2"

   Segment definitions:
     - HORIZON   = horizon such as intraday, daily, morning, hourly
                   = INTRADAY, DAILY, D2, M5, OPEN, EOD
     - ARCHETYPE = archetype of market behavior
                   = REVERSION, MOMENTUM, BREAKOUT, ARBITRAGE, SCALPING
     - ENTRY     = primary entry trigger or catalyst
                   = VWAP, GAP, PB, CO, THRUST, NEWS
     - VARIANT   = (optional, one or multiple terms) contextual or structural qualifiers
                   = e.g., LOWVOL, PAIR, EOD, CMP, VOLSTRONG
     - VERSION   = (optional) version or lifecycle stage
                   = V1, V2, XP, LIVE, BETA
     - Future extensions: VARIANT segments will become mandatory, such as CONTEXT, RISK, ASSET, etc., in specific ordering

2. Strategy Class Name
   - Format: PascalCase + "Strategy"
   - Example: DailyReversionVwapV1Strategy
   - Must inherit from BaseStrategy
   - STRATEGY_CODE should be defined as a class-level constant

3. Python File Name
   - Format: strategy_[strategy_code_lowercase].py
   - Example: strategy_daily_reversion_vwap_v1.py

Example 1:
    STRATEGY_CODE:     "DAILY_REVERSION_VWAP_V1"
    Class Name:        DailyReversionVwapV1Strategy
    File Name:         strategy_daily_reversion_vwap_v1.py

Example 2:
    STRATEGY_CODE:     "INTRADAY_BREAKOUT_GAP_CMP_VOLSTRONG_V2"
    Class Name:        IntradayBreakoutGapCmpVolstrongV2Strategy
    File Name:         strategy_intraday_breakout_gap_cmp_volstrong_v2.py

Example 3:
    STRATEGY_CODE:     "DAILY_REVERSION_CO_PAIR_V2"
    Class Name:        DailyReversionCoPairV2Strategy
    File Name:         strategy_daily_reversion_co_pair_v2.py

"""
from datetime import datetime, time, timedelta
import backtrader as bt
from .dnse_utils import setup_logger
from .position_manager import PositionManager
from .bar_aggregator import BarAggregator

class ConditionFlag:
    """
    A boolean flag with time-based validity, used for caching logical conditions in layered strategy logic.

    Core behavior:
    - Stores a boolean value along with an expiry timestamp
    - update(), set_true(), and set_false() set the value and refresh validity
    - invalidate() marks the flag as invalid
    - If recompute_fn is provided, value() auto-updates when expired

    Usage modes:
    - Manual mode:
        if not flag.is_valid():
            flag.update(compute_value())
        if flag.value(): ...

    - Auto-recompute mode:
        if flag.value(): ...
    
    Future extensions:
    - FlagManager for centralized flag management
    - Adaptive expiry via volatility or regime context
    """

    # when needed, use this constant for expiry_seconds in update() to force a fast recompute
    FAST_RECOMPUTE_SECONDS = 1

    def __init__(self, strategy, expiry_seconds=300, recompute_fn=None):
        self.dt = strategy.data.datetime
        self.expiry_duration = timedelta(seconds=expiry_seconds)
        self.recompute_fn = recompute_fn # (optional) callback function to recompute value
        self._value = False
        self.set_bar_idx = 0 # bar index when flag set
        self.expiry = datetime.min
    
    def is_valid(self):
        return self.dt.datetime() <= self.expiry
    
    def update(self, value, expiry_seconds=None):
        """Update the flag value and expiry. Optionally override expiry duration."""
        self._value = value
        self.set_bar_idx = len(self.dt)
        self.expiry = self.dt.datetime() + (self.expiry_duration if expiry_seconds is None else timedelta(seconds=expiry_seconds))

    def set_true(self, expiry_seconds=None):
        """Update the value to True"""
        self.update(True, expiry_seconds)
    
    def set_false(self, expiry_seconds=None, recompute_next_bar=False):
        """Update the value to False, optionally forcing a fast recompute on next access"""
        self.update(False, self.FAST_RECOMPUTE_SECONDS if recompute_next_bar else expiry_seconds)
    
    def invalidate(self):
        """Invalidate the flag, forcing recomputation on next access"""
        self.expiry = datetime.min

    def value(self):
        if not self.is_valid():
            if self.recompute_fn:
                value = self.recompute_fn()
                self.update(value)
            else: # without recompute function, should not call value() when not is_valid()
                raise ValueError("Flag has no recompute function and is not manually set")
        return self._value
    
class CompositeScore:
    """Register and log scores for a composite condition"""

    def __init__(self, strategy, *, prefix="score"):
        self.strategy = strategy
        self.prefix = prefix
        self.score = 0
        self.max_score = 0
    
    def add(self, name, value, max_score=1):
        score = max_score if value else 0
        self.strategy.log(f"{self.prefix}_{name}", score)
        self.score += score
        self.max_score += max_score
        
    def get_score(self):
        self.strategy.log(f"{self.prefix}", self.score)
        self.strategy.log(f"max_{self.prefix}", self.max_score)
        return self.score
    
    def get_score_ratio(self):
        score_ratio = self.score / max(self.max_score, 1e-6)
        self.strategy.log(f"{self.prefix}", self.score)
        self.strategy.log(f"max_{self.prefix}", self.max_score)
        self.strategy.log(f'{self.prefix}_ratio', score_ratio)
        return score_ratio
    
class BaseStrategy(bt.Strategy):
    """
    Base class for layered logic strategies
    """
    CONFIG = {
        "default": {}
    }
    
    params = (
        ('regime', "default"),
        
        # minimum data length
        ('min_data_len_n_days', 1),

        # time
        ('day', [0, 1, 2, 3, 4]),             # Days to trade (0=Mon ... 4=Fri)
        ('entry_fromtime', time(9, 30)),      # Earliest time to consider entry
        ('entry_totime', time(14, 28)),       # Latest time to consider entry
        ('exit_fromtime', time(9, 0)),        # Do not allow exit before this time
        ('exit_totime', time(14, 30)),        # Last time to allow natural exit
        ('min_hold_bars', 5),                 # Minimum holding bars
        ('force_exit_time', time(14, 28)),    # Forcefully exit all positions at or after this time

        # entry
        ('entry_ma_fast', 1),
        ('entry_ma_slow', 1),

        # exit
        ('exit_ma_fast', 1),
        ('exit_ma_slow', 1),

        # risk/position management
        ('max_trades_per_day', 1),
        ('size', 10),
        ('sl_ratio', 0.01), # ratio of entry price
        ('tp_ratio', 0.05), # ratio of entry price
        ('trailing_stop', True),
    )
    
    def set_params(self, **params):
        """Update or add new parameters to self.params"""
        for k, v in params.items():
            setattr(self.params, k, v)

    def __init__(self):
        self.set_params(**self.CONFIG[self.p.regime])
        super().__init__()
        self.logger = setup_logger(name=self.__class__.__name__)
        self.bar_agg = BarAggregator(self.data)
        self.min_data_len_condition = False # if there enough data to start trading
        self.logs = {}
        self.position_manager = PositionManager(self)
        self.position_manager.enable_sl_tp(sl_ratio=self.p.sl_ratio, tp_ratio=self.p.tp_ratio, trailing_stop=self.p.trailing_stop)
        self.trade_direction = None
        self.position_size_multiplier = 1
        
        ##############################################
        # entry pre-condition
        self.daily_trade_count = 0

        # entry context
        self.entry_context_flag = ConditionFlag(self)

        # entry setup
        self.entry_setup_flag = ConditionFlag(self)

        # entry signal
        self.entry_ma_fast = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.entry_ma_fast)
        self.entry_ma_slow = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.entry_ma_slow)

        # entry execution
        self.entry_bar_idx = None
        self.entry_datetime = None

        ##############################################
        # exit pre-condition

        # exit context

        # exit setup

        # exit signal
        self.exit_ma_fast = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.exit_ma_fast)
        self.exit_ma_slow = bt.indicators.ExponentialMovingAverage(self.data.close, period=self.p.exit_ma_slow)

        # exit execution

    def entry_pre_condition(self):
        # check minimum data length
        if not self.min_data_len_condition:
            d = self.bar_agg.get_prev_n_days_bar(max(self.p.min_data_len_n_days, 1))
            if d.coverage < 1: # if not enough data...
                return False # ...then wait for more
            else:
                self.min_data_len_condition = True
        
        # reset daily trade count if new day
        if self.entry_datetime is None or self.data.datetime.date() > self.entry_datetime.date(): # new day
            self.daily_trade_count = 0
        
        # check daily trade limit
        if self.daily_trade_count >= self.p.max_trades_per_day:
            return False
        
        return (self.data.datetime.date().weekday() in self.p.day and
                self.p.entry_fromtime <= self.data.datetime.time() <= self.p.entry_totime)

    def entry_context_condition(self):
        return True
    
    def entry_setup_condition(self):
        if not self.entry_setup_flag.is_valid():
            self.entry_setup_flag.set_true(expiry_seconds=8*3600) # 8 hours = valid for remaining trading time of the day
        return self.entry_setup_flag.value()

    def entry_signal(self):
        return self.entry_signal_ma()
           
    def evaluate_and_execute_entry(self):
        if self.position_manager.size == 0 and self.entry_pre_condition() and self.entry_context_condition() and self.entry_setup_condition() and self.entry_signal():
            if self.position_manager.enter_position(size=self.p.size, multiplier=self.position_size_multiplier, trade_direction=self.trade_direction, **self.logs) > 0:
                self.daily_trade_count += 1
                self.entry_bar_idx = len(self.data)
                self.entry_datetime = self.data.datetime.datetime()
                self.position_size_multiplier = 1
                self.entry_setup_flag.invalidate()

    def exit_pre_condition(self):
        return self.exit_pre_condition_hold_bars()

    def exit_context_condition(self):
        return True
    
    def exit_setup_condition(self):
        return True
  
    def exit_signal(self):
        return self.exit_signal_ma()

    def evaluate_and_execute_exit(self):
        if self.position_manager.size == 0:
            return
        normal_exit = self.exit_pre_condition() and self.exit_context_condition() and self.exit_setup_condition() and self.exit_signal()        
        reason = "EXIT_POSITION" if normal_exit else self.position_manager.check_sl_tp()

        if reason:
            # log before execution
            self.log('trade_direction', self.trade_direction)
            self.log("entry_bar_idx", self.entry_bar_idx)
            self.log("entry_date", self.entry_datetime.date())
            self.log("entry_time", self.entry_datetime.time())
            self.log("entry_session", "morning" if self.entry_datetime.hour < 12 else "afternoon")
            self.log('exit_bar_idx', len(self.data))
            self.log('hold_bars', len(self.data) - self.entry_bar_idx)
            self.log('hold_minutes', (self.data.datetime.datetime() - self.entry_datetime).total_seconds() / 60)

            # execution
            if self.position_manager.close_position(reason=reason, **self.logs) > 0:
                self.clear_logs()

    def next(self):
        self.evaluate_and_execute_entry()
        self.evaluate_and_execute_exit()

    def notify_order(self, order):
        self.position_manager.update(order)

    def stop(self):
        print(f"(size = {self.p.size}, sl = {self.p.sl_ratio}, tp = {self.p.tp_ratio}), num_trades = {len(self.position_manager.transaction_log)}, value: {self.broker.getvalue():,.0f}")
        self.logger.info("Saving status... TBD")
        self.logger.info('Strategy stopped')

    def log(self, key, value):
        self.logs[key] = value

    def log_dict(self, log_dict):
        self.logs.update(log_dict)

    def log_bar(self, bar, prefix=None):
        self.log_dict(bar.to_dict(prefix))
    
    def clear_logs(self):
        self.logs = {}

    ##############################################
    # Specialized functions for derived strategies to use
    ##############################################
    
    def entry_context_condition_d10_regime(self):
        if not self.entry_context_flag.is_valid():
            regime = self.bar_agg.get_regime_d10()
            self.log_dict(regime)
            if regime['regime.id'] == 0:
                self.entry_context_flag.set_false(expiry_seconds=12*3600) # 12 hours = valid for full trading day
            else:
                self.entry_context_flag.set_true(expiry_seconds=12*3600) # 12 hours = valid for full trading day
        return self.entry_context_flag.value()
    
    def entry_signal_ma(self):
        """Entry signal based on MA crossover"""
        if self.p.entry_ma_fast == self.p.entry_ma_slow: # disabled
            return True
        if self.trade_direction == "LONG":
            return self.entry_ma_fast[0] > self.entry_ma_slow[0] # on trend
        elif self.trade_direction == "SHORT":
            return self.entry_ma_fast[0] < self.entry_ma_slow[0] # on trend
        return True
    
    def exit_pre_condition_hold_minutes(self):
        return (self.data.datetime.time() >= self.p.force_exit_time or
                (self.p.exit_fromtime <= self.data.datetime.time() <= self.p.exit_totime and
                self.data.datetime.datetime() >= self.entry_datetime + timedelta(minutes=self.p.min_hold_minutes)))
    
    def exit_pre_condition_hold_bars(self):
        return (self.data.datetime.time() >= self.p.force_exit_time or
                (self.p.exit_fromtime <= self.data.datetime.time() <= self.p.exit_totime and
                len(self.data) >= self.entry_bar_idx + self.p.min_hold_bars))
        
    def exit_signal_ma(self):
        """Exit signal based on MA crossover"""
        if self.data.datetime.time() >= self.p.force_exit_time:
            return True
        if self.p.exit_ma_fast == self.p.exit_ma_slow: # disabled
            return True
        if self.trade_direction == "LONG": 
            return self.exit_ma_fast[0] < self.exit_ma_slow[0] # trend exhausted
        elif self.trade_direction == "SHORT": 
            return self.exit_ma_fast[0] > self.exit_ma_slow[0] # trend exhausted
        return True
