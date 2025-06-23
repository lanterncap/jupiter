import pandas as pd
import numpy as np
from datetime import datetime, date, time
import backtrader as bt
from backtrader.metabase import MetaParams
from backtrader.feed import DataBase
from backtrader.utils.py3 import queue, with_metaclass
from backtrader.utils import date2num
import itertools
import matplotlib.pyplot as plt
from jupiter.dnse_utils import VN30F1MCommission, current_date_in_vietnam
from jupiter.analyzers import TimeAnalyzer
from jupiter.dnse_utils import setup_logger, BAR_TEMPLATE

class MetaSingleton(MetaParams):
    def __init__(cls, name, bases, dct):
        super(MetaSingleton, cls).__init__(name, bases, dct)
        cls._singleton = None

    def __call__(cls, *args, **kwargs):
        if cls._singleton is None:
            cls._singleton = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._singleton
    
class BacktestStore(with_metaclass(MetaSingleton, object)):
    BrokerCls = None
    DataCls = None

    @classmethod
    def getdata(cls, *args, **kwargs):
        return cls.DataCls(*args, **kwargs)

    @classmethod
    def getbroker(cls, *args, **kwargs):
        return cls.BrokerCls(*args, **kwargs)

    def __init__(self, **kwargs):
        super(BacktestStore, self).__init__()
        self.logger = setup_logger(name=self.__class__.__name__)
        self.notifs = queue.Queue()  # Notifications for cerebro
        self._env = None  # Reference to cerebro
        self.broker = None  # Broker instance
        self.datas = list()  # Data feeds that have registered
        
    def put_notification(self, msg, *args, **kwargs):
        self.notifs.put((msg, args, kwargs))

    def get_notifications(self):
        """Return the pending store notifications"""
        self.notifs.put(None)  # Mark end of current notifications
        notifs = list()
        while True:
            notif = self.notifs.get()
            if notif is None:  # Reached marker
                break
            notifs.append(notif)
        return notifs

    def start(self, data=None, broker=None):
        """Called by either data feed or broker"""
        if data is not None:
            self._env = data._env
            self.datas.append(data)

        if broker is not None:
            self.broker = broker

    def stop(self):
        while not self.notifs.empty():
            try:
                self.notifs.get_nowait()
            except queue.Empty:
                break

class MetaBacktestData(DataBase.__class__):
    def __init__(cls, name, bases, dct):
        super(MetaBacktestData, cls).__init__(name, bases, dct)
        BacktestStore.DataCls = cls

class BacktestData(with_metaclass(MetaBacktestData, DataBase)):
    params = (
        ('name', "VN30F1M"),
        ('historical', False),  # False (default) = continue with live data; True = will stop after loading historical data
        ('hist_datafile', 'vn30f1m-m1.csv'), # csv file contains historical data
        ('fromdate', date.min), # read historical data from date...
        ('todate', date.max), # ... to date
        ('simulate_ato', False), # simulate ATO session for historical data
    )

    # Add custom notification types
    (LIVE, CONNBROKEN, DISCONNECTED, CONNECTED, DELAYED,  # From base class
     ATO, CONTINUOUS, NOON_BREAK, ATC, CLOSED) = range(10)  # Custom notifications for market sessions

    # States of the feed
    _ST_HISTORBACK, _ST_LIVE, _ST_OVER = range(3)

    def islive(self):
        return True

    def isinlivemode(self):
        return self._state == self._ST_LIVE

    def haslivedata(self):
        return self.isinlivemode() and not self.q.empty()
    
    def getname(self):
        return getattr(self, '_name', self.p.name)

    def __init__(self, **kwargs):
        self.logger = setup_logger(name=self.__class__.__name__)
        self._store = BacktestStore(**kwargs) # store is singleton -> access to store instance
        self.session = self.CLOSED # current market session

    def setenvironment(self, env):
        '''Register _store to the env (the cerebro) so that get_notifications is called and
        notifs are propagated to notify_store in each strategy object'''
        super(BacktestData, self).setenvironment(env)
        env.addstore(self._store)

    def start(self):
        super(BacktestData, self).start()
        self._store.start(data=self) # bind with the store

        # start in historical data mode
        self._state = self._ST_HISTORBACK
        self.put_notification(self.DELAYED)

        self.q = queue.Queue() # queue for both historical and live data bars
        self._last_bar_dt = None # datetime of last bar, track to ignore seen bar

        # load historical data
        df = pd.read_csv(self.p.hist_datafile)
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['hour'])
        df = df[['datetime', 'open', 'high', 'low', 'close', 'volume']]
        df = df[(df['datetime'] >= self.p.fromdate) & (df['datetime'] <= self.p.todate)]        
        for _, row in df.iterrows():
            bar = BAR_TEMPLATE.copy()
            bar['datetime'] = row['datetime'].to_pydatetime() # row['datetime'] is a reference, convert to datetime object
            bar['open'] = float(row['open'])
            bar['high'] = float(row['high'])
            bar['low'] = float(row['low'])
            bar['close'] = float(row['close'])
            bar['volume'] = int(row['volume'])

            # FOR BACKTESTING - STEP 1 of 2: insert a synthesized ato bar before the 9:00 bar
            if (self.p.simulate_ato and bar['datetime'].hour, bar['datetime'].minute) == (9, 0):
                ato_bar = BAR_TEMPLATE.copy()
                ato_bar['datetime'] = datetime.combine(bar['datetime'].date(), time(8, 45))
                ato_bar['open'] = ato_bar['high'] = ato_bar['low'] = ato_bar['close'] = bar['open'] # NOTE: alternatively, float('nan')?
                ato_bar['volume'] = 0
                self.q.put(ato_bar)

            self.q.put(bar)
        if self.q.empty():
            print(f"No historical data found for {self.p.hist_datafile} from {self.p.fromdate} to {self.p.todate}")

    def stop(self):
        super(BacktestData, self).stop()
        self._store.stop()

    def _load(self):
        if self._state == self._ST_OVER:
            return False
        elif self._state == self._ST_LIVE:
            return self._load_bar()
        elif self._state == self._ST_HISTORBACK:
            if self._load_bar():
                # FOR BACKTESTING - STEP 2 of 2: notify ATO and ATC session
                # synthesize ATO and ATC sessions for historical data
                if (self._last_bar_dt.hour, self._last_bar_dt.minute) == (8, 45):
                    self.notify_session(self.ATO, self._last_bar_dt)
                if (self._last_bar_dt.hour, self._last_bar_dt.minute) == (14, 30):
                    self.notify_session(self.ATC, self._last_bar_dt)
                return True
            else:
                self._start_live()

    def _load_bar(self):
        try:
            bar = self.q.get(timeout=self._qcheck)
        except queue.Empty:
            return None
        bar_dt = bar['datetime']
        if self._last_bar_dt is not None:
            while bar_dt <= self._last_bar_dt:
                self.logger.info(f"{bar_dt}: Seen this bar!")
                try:
                    bar = self.q.get(timeout=self._qcheck)
                except queue.Empty:
                    return None
                bar_dt = bar['datetime']
        self._last_bar_dt = bar_dt # vietnam time, no tzinfo
        self.lines.datetime[0] = date2num(bar_dt)
        self.lines.open[0] = bar['open']
        self.lines.high[0] = bar['high']
        self.lines.low[0] = bar['low']
        self.lines.close[0] = bar['close']
        self.lines.volume[0] = bar['volume']
        self.lines.openinterest[0] = 0
        return True

    def _start_live(self):
        self._state = self._ST_OVER

    def notify_session(self, status, dt=None):
        self.session = status
        self.notifs.append((status, (), {'dt': dt}))
        self._laststatus = status

class VN30F1MFixedSizer(bt.Sizer):
    params = (
        ('mult', 100_000),  # Contract multiplier (VND per point)
    )
    def _getsizing(self, comminfo, cash, data, isbuy):
        return 100

def run_single_backtest(strategy_class, data_feed, initial_cash=100_000_000, params=None):
    """Run a single backtest with given parameters"""
    cerebro = bt.Cerebro()
    cerebro.adddata(data_feed)

    # add m60 data
    #data_m60_name = '_'.join(data_feed.getname().split('_')[:-1] + ['m60'])
    #cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Minutes, compression=60, name=data_m60_name)

    # add d1 data
    #data_d1_name = '_'.join(data_feed.getname().split('_')[:-1] + ['d1'])
    #cerebro.resampledata(data_feed, timeframe=bt.TimeFrame.Days, compression=1, name=data_d1_name)

    # Add strategy
    if params:
        cerebro.addstrategy(strategy_class, **params)
    else:
        cerebro.addstrategy(strategy_class)

    cerebro.addsizer(VN30F1MFixedSizer)
    cerebro.broker.addcommissioninfo(VN30F1MCommission())
    cerebro.broker.setcash(initial_cash)

    # Add analyzers
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name='trades')
    cerebro.addanalyzer(bt.analyzers.Returns, _name='returns')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Months, _name='monthly_returns')
    cerebro.addanalyzer(bt.analyzers.VWR, _name='vwr')  # Variability-Weighted Return
    cerebro.addanalyzer(TimeAnalyzer, _name='time_analysis')

    # Run backtest
    results = cerebro.run()
    strategy = results[0]

    return strategy, cerebro

def get_win_rate(trade_analysis):
    """Calculate win rate from trade analysis"""
    if not hasattr(trade_analysis, 'total'):
        return 0.0

    total = trade_analysis.total.total if hasattr(trade_analysis.total, 'total') else 0
    if total == 0:
        return 0.0

    won = trade_analysis.won.total if hasattr(trade_analysis, 'won') else 0
    return (won / total) * 100

def optimize_strategy(strategy_class, data_feed, param_grid, initial_cash=100_000_000):
    """Run optimization across parameter grid"""
    # Generate all parameter combinations
    param_combinations = [dict(zip(param_grid.keys(), v))
                          for v in itertools.product(*param_grid.values())]
    
    # Do not run optimization if just one or no parameter combinations
    if len(param_combinations) == 0:
        return None, None
    elif len(param_combinations) == 1:
        return None, param_combinations[0]

    # Find best parameters based on lowest drawdown
    print("Running parameter optimization...")
    results = []
    best_drawdown = None
    best_params = None
    for params in param_combinations:
        strategy, cerebro = run_single_backtest(strategy_class, data_feed, initial_cash, params)

        # Calculate key metrics
        drawdown = strategy.analyzers.drawdown.get_analysis()
        trade_analysis = strategy.analyzers.trades.get_analysis()

        result = params.copy()
        result['num_trades'] = trade_analysis.total.total
        result['max_drawdown'] = drawdown.max.drawdown
        result['win_rate'] = get_win_rate(trade_analysis)
        result['accum_profit'] = strategy.broker.getvalue() - initial_cash
        results.append(result)
        print(f"Case {len(results)} of {len(param_combinations)}: {params}, accumulated profit: {result['accum_profit']:,.0f}")

        if best_drawdown is None or result['max_drawdown'] < best_drawdown:
            best_drawdown = result['max_drawdown']
            best_params = params

    df = pd.DataFrame(results)
    df.index = range(1, len(df) + 1)
    df.index.name = 'case_number'
    df.to_csv('output/optimization_results.csv')
    return df, best_params

def plot_equity_curve(trade_log_df):
    """Plot equity curve and drawdown"""
    if trade_log_df.empty:
        print("Warning: Trade log is empty")
        return
    
    cumulative_pnl = trade_log_df['pnlcomm'].cumsum()
    plt.figure(figsize=(12, 6))
    plt.plot(trade_log_df['date'], cumulative_pnl.values)
    strat_name = trade_log_df['strategy'].iloc[0]
    plt.title(f'Equity Curve - {strat_name} (Simulation date:{current_date_in_vietnam()})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative P&L (VND)')
    plt.grid(True)
    plt.xticks(rotation=45)    
    plt.tight_layout()
    plt.savefig('output/equity_curve.png')
    plt.close()

def plot_equity_curve2(trade_log_df):
    """Plot equity curve with price on secondary y-axis"""
    if trade_log_df.empty:
        print("Warning: Trade log is empty")
        return
    
    # Create figure and axis objects with a single subplot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot cumulative P&L on primary y-axis
    cumulative_pnl = trade_log_df['pnlcomm'].cumsum()
    color1 = '#1f77b4'  # Blue color for P&L
    ax1.plot(trade_log_df['date'], cumulative_pnl.values, color=color1, label='Cumulative P&L')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative P&L (VND)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    
    # Create secondary y-axis and plot price
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'  # Orange color for price
    ax2.plot(trade_log_df['date'], trade_log_df['entry_price'], color=color2, label='Price', alpha=0.7)
    ax2.set_ylabel('Price', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    # Add title and grid
    strat_name = trade_log_df['strategy'].iloc[0]
    plt.title(f'Equity Curve with Price - {strat_name} (Simulation date:{current_date_in_vietnam()})')
    ax1.grid(True, alpha=0.3)
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig('output/equity_curve_with_price.png')
    plt.close()

def plot_monthly_returns(monthly_returns):
    """Plot monthly returns heatmap"""
    returns_df = pd.DataFrame(monthly_returns.items(), columns=['Date', 'Return'])
    returns_df['Date'] = pd.to_datetime(returns_df['Date'])
    returns_df['Month'] = returns_df['Date'].dt.month
    returns_df['Year'] = returns_df['Date'].dt.year

    pivot_table = returns_df.pivot(index='Year', columns='Month', values='Return')
    plt.figure(figsize=(12, 8))
    plt.imshow(pivot_table, cmap='RdYlGn', aspect='auto')
    plt.colorbar(label='Returns')
    plt.title('Monthly Returns Heatmap')
    plt.savefig('output/monthly_returns_heatmap.png')
    plt.close()

def plot_trade_distribution(trade_log_df):
    """Plot trade P&L distribution"""
    if not trade_log_df.empty:
        plt.figure(figsize=(10, 6))
        plt.hist(trade_log_df['pnlcomm'], bins=50)
        plt.title('Trade P&L Distribution')
        plt.xlabel('P&L')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.savefig('output/trade_distribution.png')
        plt.close()

def analyze_results(strategy):
    """Analyze and print detailed backtest results"""
    drawdown = strategy.analyzers.drawdown.get_analysis()
    trades = strategy.analyzers.trades.get_analysis()
    returns = strategy.analyzers.returns.get_analysis()
    monthly_returns = strategy.analyzers.monthly_returns.get_analysis()
    time_analysis = strategy.analyzers.time_analysis.get_analysis()
    
    trade_log_df = pd.DataFrame(strategy.position_manager.transaction_log)
    trade_log_df.index = range(1, len(trade_log_df) + 1)
    trade_log_df.index.name = 'trade_num'
    trade_log_df.to_csv('output/detailed_trade_log.csv')
    
    trade_exit_log_df = pd.DataFrame(strategy.position_manager.trade_exit_log)
    trade_exit_log_df.index = range(1, len(trade_exit_log_df) + 1)
    trade_exit_log_df.index.name = 'trade_num'
    trade_exit_log_df.to_csv('output/trade_exit_log.csv')

    # Calculate weighted average of pnlcomm_avg_points using absolute position sizes as weights
    total_traded_size = trade_exit_log_df['size'].abs().sum()
    pnlcomm_avg_points = (trade_exit_log_df['pnlcomm_avg_points'] * trade_exit_log_df['size'].abs() / total_traded_size).sum()

    # ================
    # calculate Sharpe ratio
    # ================
    # Capital = max daily exposure (proxy by max gross position * price * contract value * margin ratio)
    trade_exit_log_df['notional'] = trade_exit_log_df['size'].abs() * trade_exit_log_df['entry_price'] * 100_000
    daily_capital = trade_exit_log_df.groupby('date')['notional'].max()
    capital = daily_capital.max() * 0.1848  # assume max margin requirement

    # Daily return = total pnl / capital
    daily_pnl = trade_exit_log_df.groupby('date')['pnlcomm'].sum()
    daily_returns = daily_pnl / capital

    # annualized Sharpe ratio
    excess_returns = daily_returns - (0.05 / 252)
    sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)

    # Calculate R:R ratio
    win_trades = trade_exit_log_df['pnlcomm'][trade_exit_log_df['pnlcomm'] > 0]
    loss_trades = trade_exit_log_df['pnlcomm'][trade_exit_log_df['pnlcomm'] < 0]
    rr_ratio = win_trades.mean() / -loss_trades.mean() if not win_trades.empty and not loss_trades.empty else np.nan

    # Basic Performance Metrics
    performance = [
        f'\nPerformance Summary:',
        f'Ending Value: {strategy.broker.getvalue():,.0f}',
        f'Total traded size: {total_traded_size:,.0f}',
        f'pnlcomm_avg_points: {pnlcomm_avg_points:,.2f}',
        f'Sharpe ratio (annualized, rfr=0.05): {sharpe:,.2f}',
        f'R:R ratio: {rr_ratio:,.1f}',
        f'Total Return (%): {(returns["rtot"] * 100):,.2f}',
        f'Max Drawdown (%): {drawdown.max.drawdown:,.2f}',
        f'Total Trades: {(trades.total.total if hasattr(trades.total, "total") else 0):,d}',
        f'Win Rate (%): {get_win_rate(trades):,.2f}',
        f'--------------------------------',
    ]
    
    for i in performance:
        print(i)

    pd.DataFrame([performance]).to_csv('output/performance_summary.csv')
    pd.DataFrame(monthly_returns.items(), columns=['Month', 'Return']).to_csv('output/monthly_returns.csv')

    # Create visualizations
    plot_equity_curve(trade_exit_log_df)
    plot_equity_curve2(trade_exit_log_df)
    plot_monthly_returns(monthly_returns)
    plot_trade_distribution(trade_log_df)
    
    return performance
