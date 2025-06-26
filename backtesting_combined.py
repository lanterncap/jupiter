from datetime import date
import numpy as np
from jupiter import SystemConfig, VN30F1MCommission, current_date_in_vietnam, portfolio_manager, VN30F1MSizer
from jupiter.backtesting import BacktestData
import backtrader as bt
import pandas as pd
import matplotlib.pyplot as plt

from jupiter.strategies import *
from jupiter.strategies.strategy_afternoon_reversion_d10vwap import AfternoonReversionD10VwapStrategy
from jupiter.strategies.strategy_daily_reversion_d1direction import DailyReversionD1DirectionStrategy
from jupiter.strategies.strategy_daily_reversion_d1direction_alt2 import DailyReversionD1DirectionAlt2Strategy
from jupiter.strategies.strategy_daily_reversion_d1direction_alt3 import DailyReversionD1DirectionAlt3Strategy
from jupiter.strategies.strategy_daily_reversion_d2direction import DailyReversionD2DirectionStrategy
from jupiter.strategies.strategy_daily_cont_d4 import DailyContD4Strategy
from jupiter.strategies.strategy_afternoon_long_s1 import AfternoonLongS1Strategy
from jupiter.strategies.strategy_afternoon_short_s1 import AfternoonShortS1Strategy
from jupiter.strategies.strategy_afternoon_short_s1_alt2 import AfternoonShortS1Alt2Strategy




# list of (strategy_class, size)
STRATEGIES = {
    OneTradePerDayStrategy: 0,

    # daily strategies
    DailyReversionD1DirectionStrategy: 100,
    DailyReversionD1DirectionAlt2Strategy: 100,
    DailyReversionD1DirectionAlt3Strategy: 100,
    DailyContD4Strategy: 100,
    DailyFollowD10VWAPStrategy: 75,

    # session (morning, afternoon) strategies
    AfternoonLongS1Strategy: 100,
    AfternoonShortS1Strategy: 100,
    AfternoonShortS1Alt2Strategy: 100,
    AfternoonContVwapStrategy: 100,

    # others
    VolumeSurgeMorningStrategy: 75,
    VolumeSurgeAfternoonStrategy: 100,
    IntradayBreakoutVolConsolStrategy: 100,

    #IntradayBreakoutVolConsolV2Strategy: 100,    
    #AfternoonReversionD10VwapStrategy: 100,
    #DailyReversionD2DirectionStrategy: 100,
}

SystemConfig.TRADING_MODE = SystemConfig.MODE_BACKTEST
SystemConfig.MUTE_LOGGERS = True

portfolio_max_net_exposure = 0.5
portfolio_max_reward = 0.2
portfolio_manager.activate(
    max_gross_exposure=sum(STRATEGIES.values()),
    max_net_exposure=int(sum(STRATEGIES.values()) * portfolio_max_net_exposure),
    max_reward=portfolio_max_reward
)

datafile = 'data/VN30F1M_m1_2020-02-10_to_2025-02-14.csv'
#datafile = 'vn30f1m-m1.csv'
fromdate = date(2024, 1, 1)
todate = date(2024, 12, 31)
todate = date(2024, 1, 30)
data_feed = BacktestData(historical=True, hist_datafile=datafile, fromdate=fromdate, todate=todate, timeframes=bt.TimeFrame.Minutes, compression=1)
data_feed.p.simulate_ato = True

cerebro = bt.Cerebro()
cerebro.adddata(data_feed)

# add strategies
cash = 0
text_box = ""
for strategy_class, size in STRATEGIES.items():
    if size > 0:
        params = {
            'size': size,
        }
        cerebro.addstrategy(strategy_class, **params)
        cash += size * 30_000_000
        text_box += f"{strategy_class.__name__} ({size})\n"
text_box += f"Total: {sum(STRATEGIES.values())}\n"
text_box += f"Max net exposure: {int(sum(STRATEGIES.values()) * portfolio_max_net_exposure)}\n"
text_box += f"Max reward: {portfolio_max_reward}"

cerebro.addsizer(VN30F1MSizer)
cerebro.broker.addcommissioninfo(VN30F1MCommission())
cerebro.broker.setcash(cash)

# Run the strategies
results = cerebro.run()
portfolio_manager.deactivate()
trade_logs = []
exit_logs = []

for strategy in results:
    trade_logs.extend(strategy.position_manager.transaction_log)
    exit_logs.extend(strategy.position_manager.trade_exit_log)

# Sort all trade logs by datetime after collecting them
trade_logs.sort(key=lambda x: pd.to_datetime(f"{x['date']} {x['time']}"))
trade_log_df = pd.DataFrame(trade_logs)
trade_log_df.index = range(1, len(trade_log_df) + 1)
trade_log_df.index.name = 'trade_num'
trade_log_df.to_csv(f'output/combined_transaction_log.csv')

exit_logs.sort(key=lambda x: pd.to_datetime(f"{x['date']} {x['time']}"))
exit_log_df = pd.DataFrame(exit_logs)
exit_log_df.index = range(1, len(exit_log_df) + 1)
exit_log_df.index.name = 'trade_num'
exit_log_df.to_csv(f'output/combined_trade_exit_log.csv')

# calculate Sharpe ratio
daily_returns = exit_log_df.groupby('date')['pnlcomm'].sum()
max_size = int(sum(STRATEGIES.values()) * portfolio_max_net_exposure)
capital = max_size * 1300 * 100_000 * 0.1848
Rt = daily_returns / capital
excess_return = Rt - (0.05/252) # risk-free rate = 0.05 per year of 252 days
Sharpe_ratio = excess_return.mean() / excess_return.std() * np.sqrt(252)
print(f"\nSharpe ratio: {Sharpe_ratio:.2f}\n")
text_box += f"\nSharpe ratio: {Sharpe_ratio:.2f}"

position_history = portfolio_manager.position_history
position_history_df = pd.DataFrame(position_history)
position_history_df.index = range(1, len(position_history_df) + 1)
position_history_df.index.name = 'trade_num'
position_history_df.to_csv(f'output/combined_position_history.csv')

# plot equity curve from trade_log_df
if True:
    plt.figure(figsize=(12, 6))
    cumulative_pnl = trade_log_df['pnlcomm'].cumsum()
    plt.plot(trade_log_df['date'], cumulative_pnl.values)
    plt.title(f'Cumulative PnL Over Time - Combined Strategies (Run date:{current_date_in_vietnam()})')
    plt.xlabel('Date')
    plt.ylabel('Cumulative PnL (VND)')
    plt.text(0.02, 0.98, text_box, transform=plt.gca().transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    plt.grid(True)
    plt.savefig('output/equity_curve_combined.png')
    plt.close()

# plot equity curve from trade_log_df + price line on the same plot
if True:
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot cumulative P&L
    cumulative_pnl = trade_log_df['pnlcomm'].cumsum()
    color1 = '#1f77b4'  # blue
    ax1.plot(trade_log_df['date'], cumulative_pnl.values, color=color1, label='Cumulative P&L')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative P&L (VND)', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    # Plot price on secondary axis
    ax2 = ax1.twinx()
    color2 = '#ff7f0e'  # orange
    ax2.plot(trade_log_df['date'], trade_log_df['entry_price'], color=color2, label='Entry Price', alpha=0.7)
    ax2.set_ylabel('Price', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    # Add title and annotation
    plt.title(f'Cumulative PnL with Price - Combined Strategies (Run date: {current_date_in_vietnam()})')
    ax1.grid(True, alpha=0.3)
    plt.text(0.02, 0.98, text_box, transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(facecolor='white', alpha=0.8))

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    # Improve formatting
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save and close
    plt.savefig('output/equity_curve_with_price_combined.png')
    plt.close()

# plot drawdown from trade_log_df
if True:
    trade_log_df['drawdown'] = trade_log_df['value'].cummax() - trade_log_df['value']
    plt.figure(figsize=(12, 6))
    trade_log_df.set_index('date')['drawdown'].plot()
    plt.title('Drawdown Over Time - Combined Strategies')
    plt.xlabel('Date')
    plt.ylabel('Drawdown (VND)')
    plt.grid(True)
    plt.savefig('output/drawdown_combined.png')
    plt.close()
