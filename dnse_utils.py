import logging
import logging.handlers
import os
import socket
from pathlib import Path
from datetime import datetime, date
import pytz
import backtrader as bt
import threading

class SystemConfig:
    """Global configuration for Jupiter trading system"""
    # Trading modes
    MODE_BACKTEST, MODE_LIVE = range(2)
    TRADING_MODE = MODE_BACKTEST

    # set this to True to turn all loggers off (probably when backtesting)
    MUTE_LOGGERS = False

utc = pytz.UTC
vietnam_tz = pytz.timezone('Asia/Ho_Chi_Minh')

BAR_TEMPLATE = {
    'datetime': datetime.min, # in Vietnam time, no tzinfo
    'open': 0.0,
    'high': 0.0,
    'low': 0.0,
    'close': 0.0,
    'volume': 0,
    'last_updated': datetime.min, # in Vietnam time, no tzinfo
}

def now_in_vietnam_time():
    """Now() in vietnam time, then remove tzinfo"""
    return datetime.now(tz=vietnam_tz).replace(tzinfo=None)

def zulu_to_vietnam_time(str):
    """Convert strings like '2024-12-05T07:45:00.068597Z' or '2024-12-11T04:30:03Z' to vietnam time, then remove tzinfo"""
    # preprocessing: remove the microseconds if exist
    if '.' in str:
        str = str.split('.')[0] + 'Z'

    dt = datetime.strptime(str, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=utc)
    return dt.astimezone(vietnam_tz).replace(tzinfo=None)

def timestamp_to_vietnam_time(timestamp):
    """Convert UTC timestamp like 1729842300 to vietnam time, then remove tzinfo"""
    dt = datetime.fromtimestamp(timestamp, tz=utc)
    return dt.astimezone(vietnam_tz).replace(tzinfo=None)

def current_date_in_vietnam():
    """Return current date in Vietnam timezone"""
    return datetime.now(tz=vietnam_tz).date()

def setup_logger(name='logger',
                 log_dir='logs',
                 console_level=logging.DEBUG,
                 file_level=logging.INFO,
                 initial_log=True):
    """
    Get a configured logger instance with Vietnam timezone-aware timestamps.
    Creates machine-specific log files for easier tracking across environments.
    """
    logger = logging.getLogger(name)

    # Handle muted state
    if SystemConfig.MUTE_LOGGERS:
        logger.disabled = True
        logger.setLevel(logging.CRITICAL)
        return logger

    # Reset logger state
    logger.disabled = False
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Define formatters
    def vietnam_time_format(record, datefmt=None):
        return timestamp_to_vietnam_time(record.created).strftime('%Y-%m-%d %H:%M:%S +0700')

    console_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    console_formatter.formatTime = vietnam_time_format

    detail_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s')
    detail_formatter.formatTime = vietnam_time_format

    # Set up console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)

    # Create log directory if needed
    hostname = socket.gethostname().split('.')[0]
    machine_log_dir = os.path.join(log_dir, hostname)
    Path(machine_log_dir).mkdir(parents=True, exist_ok=True)

    # Set up main file handler
    log_file = os.path.join(machine_log_dir, f'{name}_{hostname}.log')
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(detail_formatter)
    file_handler.addFilter(lambda record: record.levelno in {logging.INFO})
    logger.addHandler(file_handler)

    # Set up error file handler
    error_file = os.path.join(machine_log_dir, f'error_{name}_{hostname}.log')
    error_handler = logging.handlers.RotatingFileHandler(
        error_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=5
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(detail_formatter)
    logger.addHandler(error_handler)

    # Initial log message
    if initial_log:
        logger.info(f"Logger initialized for {name} on {hostname}")

    return logger

class PrintOnce():
    """A callable that prints a message only once, useful for debugging"""
    def __init__(self):
        self.printed = set()

    def __call__(self, message, key=None):
        """
        Print a message only once.
        If key is provided, the message is printed only once for each key.
        """
        track_key = key or str(message)
        if track_key not in self.printed:
            print(message)
            self.printed.add(track_key)
    
    def reset(self):
        self.printed.clear()

print_once = PrintOnce()

def get_csv_data(dataname, fromdate, todate, compression):
    return bt.feeds.GenericCSVData(
        dataname=dataname,
        fromdate=fromdate,
        todate=todate,
        nullvalue=0.0,
        dtformat='%Y-%m-%d',
        tmformat='%H:%M:%S',
        datetime=0,
        time=1,
        open=2,
        high=3,
        low=4,
        close=5,
        volume=6,
        openinterest=-1,
        timeframe=bt.TimeFrame.Minutes,
        compression=compression,
        headers=True
    )

def get_m1_data(dataname='amix-vn30f1m-m1-2024-07-19-to-2024-11-13.csv',
                fromdate=date(2024, 6, 1),
                todate=date(9999, 1, 1)):
    return get_csv_data(dataname, fromdate, todate,1)

def get_m5_data(dataname='amix-vn30f1m-m5-2020-08-03-to-2024-07-05.csv',
                fromdate=date(2024, 6, 1),
                todate=date(9999, 1, 1)):
    return get_csv_data(dataname, fromdate, todate, 5)

def get_vn30f1m_trading_symbol(trading_date=None):
    """
    Returns the VN30F1M futures symbol that is actively trading for a given trading date.

    Symbol convention:
    ------------------
    1. **Old system (pre-KRX, until June 2025 expiry):**
       Format: VN30FYYMM
       - YY: Last two digits of the expiry year.
       - MM: Two-digit expiry month.
       Example: May 2025 expiry → 'VN30F2505'

    2. **New KRX system (from July 2025 expiry onward):**
       Format: 41I1<YearCode><MonthCode>000
       - 4   : Derivative product
       - 1   : Future
       - I1  : Underlying asset (VN30 Index)
       - <YearCode>: Encoded expiry year (F = 2025, G = 2026, etc.)
       - <MonthCode>: Encoded expiry month (1 = Jan, ..., 9 = Sep, A = Oct, B = Nov, C = Dec)
       - 000 : Standardized contract suffix for VN30 futures
       Example: May 2025 expiry → '41I1F5000'

    Expiry logic:
    -------------
    - VN30F1M contracts expire on the **third Thursday** of each month.
    - If the trading date is **after** the expiry date for the current month,
      the symbol moves to the **next month's contract**.

    Args:
        trading_date (datetime.date, optional): The date for which to determine the trading symbol.
                                                If None, uses the current Vietnam date.

    Returns:
        str: The active VN30F1M futures contract symbol for the given trading date.

    Raises:
        ValueError: If the expiry year is outside the supported KRX mapping range (2020–2039).

    Example:
        >>> get_vn30f1m_trading_symbol(date(2025, 5, 10))
        'VN30F2505'

        >>> get_vn30f1m_trading_symbol(date(2025, 7, 10))
        '41I1F7000'
    """
    trading_date = trading_date or current_date_in_vietnam()
    
    def third_thursday(year, month):
        """Find third Thursday of the month."""
        # Get first day of month
        first_day = date(year, month, 1)
        
        # Calculate first Thursday (3 is Thursday in weekday())
        first_thursday = 1 + (3 - first_day.weekday() + 7) % 7
        
        # Add 14 days to get to third Thursday
        third_thursday_date = date(year, month, first_thursday + 14)
        return third_thursday_date

    # determine contract month
    year = trading_date.year
    month = trading_date.month
    expiry = third_thursday(year, month)
    
    if trading_date > expiry:
        # move to next month's contract
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    # old style cut-off: VN30F2506 = June 2025 still old
    if (year < 2025) or (year == 2025 and month <= 6):
        # Old style VN30FYYMM
        symbol_year = year % 100
        symbol = f"VN30F{symbol_year:02d}{month:02d}"
    else:
        # KRX new style
        asset_code = "I1"  # Asset: VN30

        year_map = {
            2020: 'A', 2021: 'B', 2022: 'C', 2023: 'D', 2024: 'E',
            2025: 'F', 2026: 'G', 2027: 'H', 2028: 'J', 2029: 'K',
            2030: 'L', 2031: 'M', 2032: 'N', 2033: 'P', 2034: 'Q',
            2035: 'R', 2036: 'S', 2037: 'T', 2038: 'V', 2039: 'W'
        }
        if year not in year_map:
            raise ValueError(f"Year {year} not supported for KRX mapping.")
        year_code = year_map[year]

        month_map = {
            1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6',
            7: '7', 8: '8', 9: '9', 10: 'A', 11: 'B', 12: 'C'
        }
        month_code = month_map[month]

        symbol = f"41{asset_code}{year_code}{month_code}000"
        
    return symbol

def get_vn30f1m_trading_symbol_old(trading_date=None): # old style. TODO: to remove this function
    """
    Returns the VN30F1M futures symbol that is actively trading for a given date.
    VN30F1M futures contracts expire on the third Thursday of each month.

    Args:
        trading_date (date): The date for which to get the VN30F1M futures symbol.
                             If None, uses the current date in Vietnam timezone.

    Returns:
        str: The VN30F1M futures contract symbol (e.g., "VN30F2411" for November 2024)

    Example usage:
        # Get symbol for current Vietnam date
        symbol = get_vn30f1m_trading_symbol()

        # Get symbol for specific date
        from datetime import date
        d = date(2024, 10, 22)
        symbol = get_vn30f1m_trading_symbol(d)
    """
    trading_date = trading_date or current_date_in_vietnam()
    
    def third_thursday(year, month):
        # Get first day of month
        first_day = date(year, month, 1)
        
        # Calculate first Thursday (3 is Thursday in weekday())
        first_thursday = 1 + (3 - first_day.weekday() + 7) % 7
        
        # Add 14 days to get to third Thursday
        third_thursday_date = date(year, month, first_thursday + 14)
        return third_thursday_date

    year = trading_date.year
    month = trading_date.month
    third_thursday_current = third_thursday(year, month)

    # If current date is past the third Thursday, move to the next month
    if trading_date > third_thursday_current:
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1

    symbol_year = year % 100  # Get last two digits of the year
    symbol = f"VN30F{symbol_year:02d}{month:02d}"
    return symbol

class VN30F1MCommission(bt.CommInfoBase):
    """
    Commission scheme for VN30F1M futures. It has two parts: fixed, plus percentage
    - VN30 is an index; VN30F1M is the futures contract
    - Contract value = VND 100,000 per index point
    - Fixed part: 2700 VND per contract
    - Variable part: (0.1% / 2) of 17% contract value
    - Margin requirement: 18.48% of contract value
    """
    params = (
        ('commission_fixed', 2700+2550),  # Base commission in VND
        ('commission_pct', 0.001/2*0.17),  # Variable commission rate
        ('stocklike', False),  # indicate this is a futures instrument
        ('mult', 100_000),  # Contract multiplier (VND per point)
        ('margin', 0.1848),  # Margin requirement as percentage
        ('commtype', bt.CommInfoBase.COMM_FIXED),  # Commission type; could remove but keep here for completeness
    )

    def _getcommission(self, size, price, pseudoexec):
        """Calculate commission for the trade"""
        contract_value = price * self.p.mult
        commission = self.p.commission_fixed + contract_value * self.p.commission_pct
        return abs(size) * commission

    def get_margin(self, price):
        """Calculate required margin"""
        return price * self.p.mult * self.p.margin

class VN30F1MSizer(bt.Sizer):
    """
    Position sizer for VN30F1M futures
    """
    params = (
        ('margin', 0.1848),  # margin requirement
        ('mult', 100_000),  # Contract multiplier (VND per point)
    )

    def _getsizing(self, comminfo, cash, data, isbuy):
        """
        Calculate the number of contracts to buy or sell based on the available cash and the margin requirement.
        Called by strategy.buy() or strategy.sell() when no size is specified
        """
        price = data.close[0]
        value_per_contract = price * self.p.mult
        margin_per_contract = value_per_contract * self.p.margin
        return cash // margin_per_contract
    
class GoogleSheetLogger:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._instance_lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized"):
            return
        self._initialized = True
        self._active = False

    def activate(self): self._active = False
    def deactivate(self): self._active = False
    def shutdown(self): pass

    def log_message(self, sheet_name, row_data): pass
    def log_value(self, value): pass
    def log_error(self, message): pass
    def log_info(self, message): pass
    def log_order(self, order, pm_gross=0, pm_net=0): pass

gsheet_logger = GoogleSheetLogger()

class Arbiter:
    _instance = None
    _instance_lock = threading.Lock()

    def __new__(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if hasattr(self, "_initialized") and self._initialized:
            return
        self._initialized = True
        self._active = False

    def activate(self): self._active = False
    def deactivate(self): self._active = False
    def shutdown(self): pass

    def get_param(self, owner, param, param_type=str, default_value=None):
        return default_value

arbiter = Arbiter()
