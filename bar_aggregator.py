from datetime import datetime, time
from functools import cached_property

class AggregatedBar:
    """
    Synthetic OHLCV bar with lazily computed properties, returned by BarAggregator methods
    
    Fields:
    - open, high, low, close, volume
    - CO: close - open
    - HL: high - low
    - direction: 'UP', 'DOWN', or 'FLAT'
    - top_wick, bottom_wick, body_strength: as ratios of HL
    - directional_volume: net sum of bar volumes, signed by each bar's direction (+volume if close > open, -volume if close < open)
    - from_datetime, to_datetime: datetime range of the aggregated bar
    """
    
    def __init__(self, open_, high, low, close, volume, directional_volume, vwap, from_datetime, to_datetime, info=None):
        self.open = open_
        self.high = high
        self.low = low
        self.close = close
        self.volume = volume
        self.directional_volume = directional_volume
        self.vwap = vwap
        self.vwap_deviation = close - vwap
        self.from_datetime = from_datetime
        self.to_datetime = to_datetime
        self.info = info or {}

    @cached_property
    def CO(self):
        return self.close - self.open

    @cached_property
    def HL(self):
        return self.high - self.low

    @cached_property
    def direction(self):
        return 'UP' if self.CO > 0 else ('DOWN' if self.CO < 0 else 'FLAT')

    @cached_property
    def directional_volume_ratio(self):
        return self.directional_volume / max(self.volume, 1e-6)
    
    @cached_property
    def vwap_side(self):
        return 'ABOVE' if self.vwap_deviation > 0 else ('BELOW' if self.vwap_deviation < 0 else 'AT')

    @cached_property
    def top_wick(self): # as ratio of HL
        return (self.high - max(self.open, self.close)) / max(self.HL, 1e-6)

    @cached_property
    def bottom_wick(self): # as ratio of HL
        return (min(self.open, self.close) - self.low) / max(self.HL, 1e-6)

    @cached_property
    def body_strength(self): # as ratio of HL
        return abs(self.CO) / max(self.HL, 1e-6)
    
    def get_info(self, key, default=None):
        return self.info.get(key, default)
    
    @cached_property
    def coverage(self):
        return self.get_info('coverage', 0)
        
    def to_dict(self, prefix=None):
        """Return a dict representation of the aggregated bar, for easy logging"""
        if prefix:
            prefix = f'{prefix}.'
        return {
            f'{prefix}open': self.open,
            f'{prefix}high': self.high,
            f'{prefix}low': self.low,
            f'{prefix}close': self.close,
            f'{prefix}volume': self.volume,
            f'{prefix}CO': self.CO,
            f'{prefix}HL': self.HL,
            f'{prefix}direction': self.direction,
            f'{prefix}top_wick': self.top_wick,
            f'{prefix}bottom_wick': self.bottom_wick,
            f'{prefix}body_strength': self.body_strength,
            f'{prefix}directional_volume': self.directional_volume,
            f'{prefix}directional_volume_ratio': self.directional_volume_ratio,
            f'{prefix}vwap': self.vwap,
            f'{prefix}vwap_deviation': self.vwap_deviation,
            f'{prefix}vwap_side': self.vwap_side,
            f'{prefix}from_datetime': self.from_datetime,
            f'{prefix}to_datetime': self.to_datetime,
            f'{prefix}coverage': self.coverage,
        }

class BarAggregator:
    """
    Utility class for aggregating OHLCV bars

    Features:
    - Flexible datetime-based aggregation for synthetic bars
    - Relative time/date calculations

    Assumptions:
    - data is a backtrader data feed, sorted by datetime, no duplicate timestamps
    - data.datetime, data.high, data.low, data.open, data.close, data.volume are backtrader's line objects
    - data.datetime.datetime(), data.datetime.date(), data.datetime.time() are datetime, date, time objects
    - Index convention: 0 = current bar, 1 = previous, 2 = two bars ago, etc. Note: vs backtrader's: 0 = current bar, -1 = previous, etc.
    - Aggregate(from_idx, to_idx): accept indices in any order, e.g. both (1, 5) and (5, 1) are valid; results are always computed chronologically
    - Searching for a missing bar: skip to the next available bar with a later datetime. Example: aggregating from 10:00 to 10:05 
      but 10:00 bar missing ⇒ aggregate from 10:01 to 10:05. Example: looking for previous day's last bar, but no prior 
      data exists ⇒ point to the first bar of current day
    """
    
    def __init__(self, data):
        self.data = data

    def _aggregate_result(self, open, high, low, close, volume, directional_volume, vwap, from_datetime, to_datetime, info=None):
        return AggregatedBar(open, high, low, close, volume, directional_volume, vwap, from_datetime, to_datetime, info)

    def _aggregate_null_bar(self, price, dt, info=None):
        return self._aggregate_result(open=price, high=price, low=price, close=price, volume=0, directional_volume=0, vwap=price, from_datetime=dt, to_datetime=dt, info=info)

    def _search_idx(self, target_datetime, *, side='right'):
        """Search for the index of target_datetime. If not found, return the index on the side (left or right) of the target_datetime

        Side and index convention: 
        - Left-to-right: older time is on the left side, newer time is on the right side
        - Index is positive: 0 = current bar, 1 = previous bar, etc.
        - If side = left, return the newest bar with dt <= target_datetime
        - If side = right (default), return the oldest bar with dt >= target_datetime
        - Return None if no bar found
        
        Example:
        - Bars (oldest → newest): [10:05 (4), 10:06 (3), 10:08 (2), 10:09 (1), 10:10 (0)]
        - Search for 10:06 => return 3 (exact match)
        - Search for 10:07: missing. If side = left => return 3; if side = right => return 2
        """
        if len(self.data) == 0 or side not in ('left', 'right'):
            return None

        if side == "left":
            for i in range(0, len(self.data)):
                if self.data.datetime.datetime(-i) <= target_datetime:
                    return i # return as soon as finding dt <= target_datetime
            return None # return None: not found
        else: # rìght
            if self.data.datetime.datetime() < target_datetime:
                return None
            for i in range(1, len(self.data)):
                if self.data.datetime.datetime(-i) < target_datetime: # strictly less than
                    return i-1  # step back to be >= target_datetime
            return len(self.data)-1

    def index_at_or_before(self, target_datetime):
        """Find the index of the first bar at or before target_datetime"""
        return self._search_idx(target_datetime, side='left')
    
    def index_at_or_after(self, target_datetime):
        """Find the index of the first bar at or after target_datetime"""
        return self._search_idx(target_datetime, side='right')
    
    def find_first_index_of_day(self, target_date):
        """Find the index of the first bar of the target date"""
        if len(self.data) == 0:
            return None
        if self.data.datetime.date() < target_date:
            return None
        for i in range(1, len(self.data)):
            if self.data.datetime.date(-i) < target_date:
                return i-1
        return len(self.data)-1
    
    def find_last_index_of_day(self, target_date):
        """Find the index of the last bar of the target date"""
        if len(self.data) == 0:
            return None
        for i in range(len(self.data)):
            if self.data.datetime.date(-i) == target_date:
                return i
        return None
        
    # --- Core aggregation API ---

    def aggregate_by_index(self, from_idx, to_idx=0, info=None):
        """Aggregate bars between two indices"""
        from_idx, to_idx = max(from_idx, to_idx), min(from_idx, to_idx)
        bars_to_agg = range(to_idx, from_idx + 1)
        open = self.data.open[-from_idx]
        high = max(self.data.high[-i] for i in bars_to_agg)
        low = min(self.data.low[-i] for i in bars_to_agg)
        close = self.data.close[-to_idx]
        volume = sum(self.data.volume[-i] for i in bars_to_agg)
        directional_volume = sum(
            self.data.volume[-i] if self.data.close[-i] > self.data.open[-i] else (-self.data.volume[-i] if self.data.close[-i] < self.data.open[-i] else 0)
            for i in bars_to_agg
        )

        # VWAP: sum((H+L+C)/3 * volume) / total volume
        vwap_numerator = sum(
            ((self.data.high[-i] + self.data.low[-i] + self.data.close[-i]) / 3.0) * self.data.volume[-i]
            for i in bars_to_agg
        )
        vwap = vwap_numerator / max(volume, 1e-6)

        from_datetime = self.data.datetime.datetime(-from_idx)
        to_datetime = self.data.datetime.datetime(-to_idx)
        return self._aggregate_result(open, high, low, close, volume, directional_volume, vwap, from_datetime, to_datetime, info)

    def aggregate_by_datetime(self, from_datetime, to_datetime=None, info=None):
        """Aggregate bars between two datetimes"""
        current_datetime = self.data.datetime.datetime()
        from_datetime = min(from_datetime, current_datetime)
        to_datetime = min(to_datetime or current_datetime, current_datetime)
        from_idx = self.index_at_or_after(from_datetime)
        to_idx = self.index_at_or_before(to_datetime)
        return self.aggregate_by_index(from_idx, to_idx, info)

    def aggregate_by_time(self, date, from_time, to_time=None, info=None):
        """Aggregate bars between two times on a specific date"""
        from_datetime = datetime.combine(date, from_time)
        to_datetime = datetime.combine(date, to_time) if to_time else None
        return self.aggregate_by_datetime(from_datetime, to_datetime, info)
    
    # --- Main functions ---
    def aggregate_prev_n_days(self, n_days, anchor_datetime=None):
        """Aggregate bars from the previous n full trading days ending at anchor_datetime (exclusive)."""
        assert n_days > 0, f"n_days must be positive, got {n_days}"
        anchor_datetime = anchor_datetime or self.data.datetime.datetime()
        anchor_date = anchor_datetime.date()
        
        seen_dates = set()
        first_idx = None
        last_idx = None

        for i in range(len(self.data)):
            bar_dt = self.data.datetime.datetime(-i)
            bar_date = bar_dt.date()

            if bar_date >= anchor_date:
                continue  # skip anchor day and future

            if last_idx is None:
                last_idx = i  # last bar before anchor

            seen_dates.add(bar_date)
            if len(seen_dates) <= n_days:
                first_idx = i  # keep going to find the earliest bar of that day
            elif len(seen_dates) > n_days:
                break

        if first_idx is None:
            bar_idx = self.find_first_index_of_day(anchor_date)
            return self._aggregate_null_bar(price=self.data.open[-bar_idx], dt=self.data.datetime.datetime(-bar_idx))
        
        # count how many distinct days are covered
        last_date = None
        day_count = 0
        for i in range(last_idx, first_idx + 1):
            date = self.data.datetime.date(-i)
            if date != last_date:
                last_date = date
                day_count += 1
        info = {
            'expected_days': n_days,
            'coverage': day_count/n_days,
        }
        return self.aggregate_by_index(first_idx, last_idx, info)
    
    def aggregate_prev_day(self, anchor_datetime=None):
        """Aggregate bars for the previous trading day (shorthand for aggregate_prev_n_days(1))"""
        return self.aggregate_prev_n_days(1, anchor_datetime)

    def aggregate_day_so_far(self, anchor_datetime=None, include_current_bar=True):
        """Aggregate bars for the day so far"""
        anchor_datetime = anchor_datetime or self.data.datetime.datetime()
        day_first_idx = self.find_first_index_of_day(anchor_datetime.date())
        return self.aggregate_by_index(day_first_idx, 0 if include_current_bar else 1)

    def aggregate_session_so_far(self, anchor_datetime=None, include_current_bar=True):
        """Aggregate bars for the session so far"""
        anchor_datetime = anchor_datetime or self.data.datetime.datetime()
        if anchor_datetime.time() < time(12, 0): # morning session => start from beginning of the day
            from_datetime = datetime.combine(anchor_datetime.date(), time(6, 0)) # step back to 6:00 to be safe
        else: # afternoon session => start from 12:00
            from_datetime = datetime.combine(anchor_datetime.date(), time(12, 0))
        session_first_idx = self.index_at_or_after(from_datetime)
        return self.aggregate_by_index(session_first_idx, 0 if include_current_bar else 1)

    def day_open_gap(self, anchor_datetime=None):
        """Calculate open gap between anchor day's open and previous day's close"""
        anchor_datetime = anchor_datetime or self.data.datetime.datetime()
        anchor_date = anchor_datetime.date()
        anchor_day_first_idx = self.find_first_index_of_day(anchor_date)
        if (anchor_day_first_idx is None or # no data
            len(self.data) <= anchor_day_first_idx + 1): # no previous day's data
            return 0
        anchor_day_open = self.data.open[-anchor_day_first_idx]
        prev_day_close = self.data.close[-(anchor_day_first_idx+1)]
        return anchor_day_open - prev_day_close

    # --- Aliases --- 
    def get_current_bar(self):
        return self.aggregate_by_index(0)
    
    def get_m0(self):
        return self.aggregate_by_index(from_idx=0, to_idx=0)
    
    def get_m1(self):
        return self.aggregate_by_index(from_idx=1, to_idx=1)
    
    def get_aggregated_bar(self, *, from_idx, to_idx):
        return self.aggregate_by_index(from_idx, to_idx)
    
    def get_higher_timeframe_bar(self, *, compression, at_idx=0):
        """
        Aggregate `compression` bars ending at `at_idx` (inclusive), and return a synthetic OHLCV bar

        Note on the distinction between get_higher_timeframe_bar and get_aggregated_bar: 
        - get_higher_timeframe_bar(compression=5, at_idx=0) aggregates 5 bars from index 4 to 0 (inclusive), vs. 
        - get_aggregated_bar(from_idx=5, to_idx=0) aggregates 6 bars from index 5 to 0 (inclusive)
        """
        assert compression > 0, f"compression must be positive, got {compression}"
        return self.aggregate_by_index(compression - 1 + at_idx, at_idx)

    def get_prev_n_days_bar(self, n_days, anchor_datetime=None):
        """Get the previous n days' bars, relative to `anchor_datetime`"""
        return self.aggregate_prev_n_days(n_days, anchor_datetime=anchor_datetime)

    def get_prev_day_bar(self, anchor_datetime=None):
        """Get the previous day's bar, relative to `anchor_datetime`"""
        return self.aggregate_prev_day(anchor_datetime=anchor_datetime)
    
    def get_d1(self):
        """Get the previous day's bar"""
        return self.aggregate_prev_day()
    
    def get_day_so_far_bar(self, *, include_current_bar=True):
        """Get the day-so-far bar"""
        return self.aggregate_day_so_far(include_current_bar=include_current_bar)
    
    def get_d0(self, *, include_current_bar=True):
        """Get the day-so-far bar"""
        return self.aggregate_day_so_far(include_current_bar=include_current_bar)
    
    def get_prev_session_bar(self, anchor_datetime=None):
        """Get the previous session's bar, relative to `anchor_datetime`"""
        anchor_datetime = anchor_datetime or self.data.datetime.datetime()
        if anchor_datetime.time() < time(12, 0): # morning session => return previous day's afternoon session
            d1 = self.get_prev_day_bar(anchor_datetime)
            from_datetime = datetime.combine(d1.to_datetime.date(), time(12, 0))
            to_datetime = d1.to_datetime
            from_idx = self.index_at_or_after(from_datetime)
            to_idx = self.index_at_or_before(to_datetime)
            return self.aggregate_by_index(from_idx, to_idx)
        else: # afternoon session => return same day's morning session
            from_datetime = datetime.combine(anchor_datetime.date(), time(6, 0))
            to_datetime = datetime.combine(anchor_datetime.date(), time(12, 0))
            from_idx = self.index_at_or_after(from_datetime)
            to_idx = self.index_at_or_before(to_datetime)
            return self.aggregate_by_index(from_idx, to_idx)
    
    def get_s1(self):
        """Get the previous session's bar"""
        return self.get_prev_session_bar()

    def get_s0(self, *, include_current_bar=True):
        """Get the session-so-far bar"""
        return self.aggregate_session_so_far(include_current_bar=include_current_bar)

    def get_day_open_gap(self):
        """Get the day open gap"""
        return self.day_open_gap()
    
    def get_day_open(self):
        """Get the day open"""
        day_open_idx = self.find_first_index_of_day(self.data.datetime.date())
        return self.data.open[-day_open_idx]
    
    # --- Facilities --- 
    def count_high_touches(self, aggregated_bar, tolerance=0.2):
        """Count the number of high touches within the aggregated bar"""
        from_idx = self.index_at_or_after(aggregated_bar.from_datetime)
        to_idx = self.index_at_or_before(aggregated_bar.to_datetime)
        high = aggregated_bar.high
        touch_count = 0
        last_touch_dt = None
        for i in range(to_idx, from_idx + 1): # count all the touches
            if abs(self.data.high[-i] - high) <= tolerance:
                touch_count += 1
                if last_touch_dt is None: # just need the last touch
                    last_touch_dt = self.data.datetime.datetime(-i)
        return touch_count, last_touch_dt
    
    def count_low_touches(self, aggregated_bar, tolerance=0.2):
        """Count the number of low touches within the aggregated bar"""
        from_idx = self.index_at_or_after(aggregated_bar.from_datetime)
        to_idx = self.index_at_or_before(aggregated_bar.to_datetime)
        low = aggregated_bar.low
        touch_count = 0
        last_touch_dt = None
        for i in range(to_idx, from_idx + 1):
            if abs(self.data.low[-i] - low) <= tolerance:
                touch_count += 1
                if last_touch_dt is None: # just need the last touch
                    last_touch_dt = self.data.datetime.datetime(-i)
        return touch_count, last_touch_dt
    
    def daily_atr(self, aggregated_bar):
        """Split a multi-day bar into daily bars, and calculate the average daily ATR"""
        if aggregated_bar.coverage < 1: # not enough data
            return None
        day_atrs = []
        day_first_idx = self.find_first_index_of_day(aggregated_bar.to_datetime.date())
        day_last_idx = self.find_last_index_of_day(aggregated_bar.to_datetime.date())
        day_bar = self.aggregate_by_index(day_first_idx, day_last_idx)
        prev_day_bar = self.get_prev_day_bar(day_bar.from_datetime)

        while day_bar.from_datetime >= aggregated_bar.from_datetime:
            if prev_day_bar.volume == 0: # not enough data
                break
            day_atr = max(
                day_bar.HL,
                abs(day_bar.high - prev_day_bar.close),
                abs(day_bar.low - prev_day_bar.close)
            )
            day_atrs.append(day_atr)

            day_bar = prev_day_bar
            prev_day_bar = self.get_prev_day_bar(day_bar.from_datetime)

        return sum(day_atrs) / len(day_atrs) if len(day_atrs) > 0 else 0
    
    def get_regime_d10(self, anchor_datetime=None):
        """Regime detection by aggregate over previous 10 days, relative to anchor_datetime
        
        Return a dict with the following keys:
        - regime.id = int, unique identifier for the regime (e.g. 0, 1, 2, etc.) with 0 = not enough data or invalid
        - regime.trend = 'UP', 'DOWN', or 'SIDEWAYS'
        - regime.volatility = 'LOW', 'MEDIUM', or 'HIGH'
        - regime.vwap_side = 'ABOVE', 'BELOW', or 'AT'
        """
        regime = {
            'regime.id': 0,
            'regime.id_str': None,
            'regime.trend': None,
            'regime.trend_value': None,
            'regime.volatility': None,
            'regime.volatility_value': None,
            'regime.volume_surge': None,
            'regime.volume_surge_value': None,
            'regime.version': 1,
        }
        d10 = self.aggregate_prev_n_days(10, anchor_datetime)
        if d10.coverage < 1: # not enough data
            return regime
        
        # trend
        regime['regime.trend_value'] = d10.vwap_deviation
        if d10.vwap_deviation > 25 and d10.direction == 'UP':
            regime['regime.trend'] = 'UP'
        elif d10.vwap_deviation < -25 and d10.direction == 'DOWN':
            regime['regime.trend'] = 'DOWN'
        else:
            regime['regime.trend'] = 'SIDEWAYS'
        
        # volatility
        atr = self.daily_atr(d10)
        regime['regime.volatility_value'] = atr
        if atr < 12:
            regime['regime.volatility'] = 'LOW'
        elif atr > 20:
            regime['regime.volatility'] = 'HIGH'
        else:
            regime['regime.volatility'] = 'MEDIUM'
        
        # volume surge
        d1 = self.get_prev_day_bar(anchor_datetime)
        volume_surge_value = d1.volume * 10 / d10.volume
        regime['regime.volume_surge_value'] = volume_surge_value
        regime['regime.volume_surge'] = 'YES' if volume_surge_value > 1.2 else 'NO'

        trend_map = {'UP': 0, 'DOWN': 1, 'SIDEWAYS': 2}
        vol_map = {'LOW': 0, 'MEDIUM': 1, 'HIGH': 2}
        volume_surge_map = {'YES': 0, 'NO': 1}

        trend_idx = trend_map.get(regime['regime.trend'])
        vol_idx = vol_map.get(regime['regime.volatility'])
        volume_surge_idx = volume_surge_map.get(regime['regime.volume_surge'])

        if None not in (trend_idx, vol_idx, volume_surge_idx):
            regime_id = 100 * trend_idx + 10 * vol_idx + volume_surge_idx + 1 # +1 to make 1-based index
            regime['regime.id'] = regime_id
            regime['regime.id_str'] = f'trend.{regime["regime.trend"]}_vol.{regime["regime.volatility"]}_volume_surge.{regime["regime.volume_surge"]}'

        return regime

################################
# Detect patterns from higher timeframe candles
# - agg_window is the number of 1-minute bars per higher timeframe candle
# - returns True if the pattern is detected, False otherwise
################################

def detect_bearish_engulfing(data, agg_window=5):

    required_bars = 2 * agg_window
    if len(data) < required_bars:
        return False  # not enough data
    
    aggregator = BarAggregator(data)

    # c0 = current candle (N-1 to 0), c1 = previous candle (2N-1 to N)
    c0 = aggregator.aggregate_by_index(agg_window - 1, 0)
    c1 = aggregator.aggregate_by_index(2 * agg_window - 1, agg_window)

    return (
        c1["direction"] == "UP" and
        c0["direction"] == "DOWN" and
        c0["open"] > c1["close"] and
        c0["close"] < c1["open"]
    )

################################
# Structural bar compression
# - Swing highs and swing lows
# - Support/resistance via swing clustering
# - Fractal pivot detection
# - ZigZag indicator
# - Renko Charts / Kagi / Point & Figure, non-time-based bars
################################

def detect_swing_points(data, depth=3, lookback=200):
    """
    Detect swing highs and lows from data feed

    A swing high is higher than `depth` bars before and after.
    A swing low is lower than `depth` bars before and after.

    Returns:
        List of dicts: {"index", "type": "high"/"low", "price"}
    """
    highs = [data.high[-i] for i in range(lookback)][::-1]
    lows = [data.low[-i] for i in range(lookback)][::-1]

    swing_points = []
    for i in range(depth, lookback - depth):
        is_high = all(highs[i] > highs[i - j] for j in range(1, depth + 1)) and \
                  all(highs[i] > highs[i + j] for j in range(1, depth + 1))
        is_low = all(lows[i] < lows[i - j] for j in range(1, depth + 1)) and \
                 all(lows[i] < lows[i + j] for j in range(1, depth + 1))
        if is_high:
            swing_points.append({"index": i, "type": "high", "price": highs[i], "absolute_index": len(data) - lookback + i})
        elif is_low:
            swing_points.append({"index": i, "type": "low", "price": lows[i], "absolute_index": len(data) - lookback + i})
    return swing_points


def extract_structural_candles(data, depth=3, lookback=200):
    """
    Extract structural legs from data feed based on swing highs/lows.

    Each leg (between swing points) becomes an adaptive bar capturing
    direction, length, volume, and OHLC.

    Returns:
        List of dicts: each representing a structural 'natural' candle
    """
    highs = [data.high[-i] for i in range(lookback)][::-1]
    lows = [data.low[-i] for i in range(lookback)][::-1]
    opens = [data.open[-i] for i in range(lookback)][::-1]
    closes = [data.close[-i] for i in range(lookback)][::-1]
    volumes = [data.volume[-i] for i in range(lookback)][::-1]

    swing_points = detect_swing_points(data, depth=depth, lookback=lookback)
    structural_candles = []

    for i in range(len(swing_points) - 1):
        start_idx = swing_points[i]['index']
        end_idx = swing_points[i + 1]['index']
        if start_idx >= end_idx:
            continue

        leg = {
            "type": "up_leg" if closes[end_idx] > opens[start_idx] else "down_leg",
            "from_index": start_idx,
            "to_index": end_idx,
            "open": opens[start_idx],
            "close": closes[end_idx],
            "high": max(highs[start_idx:end_idx + 1]),
            "low": min(lows[start_idx:end_idx + 1]),
            "volume": sum(volumes[start_idx:end_idx + 1]),
            "length": end_idx - start_idx + 1
        }
        structural_candles.append(leg)

    return structural_candles

