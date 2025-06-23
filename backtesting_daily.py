from datetime import date, time, datetime
import backtrader as bt
from jupiter import SystemConfig
from jupiter.backtesting import run_single_backtest, optimize_strategy, analyze_results, BacktestData


# Configure system
SystemConfig.TRADING_MODE = SystemConfig.MODE_BACKTEST
SystemConfig.MUTE_LOGGERS = True

#########################################
# Import strategy and set params
#########################################
from jupiter.strategies.strategy_daily_reversion_d1direction import DailyReversionD1DirectionStrategy as TestStrategy

# Define parameter grid for optimization
param_grid = {
    'size': [100],
    #'entry_ma_fast': [3, 4],
    #'entry_ma_slow': [10, 11, 12, 13, 15],
    #'weight': [0.2, 0.3, 0.4, 0.5, 0.6],
    #'consolidation_period': [20, 25, 30, 35], # number of bars in consolidation period
    #'consolidation_spread_threshold': [1.2/1200, 1.5/1200, 2/1200, 2.5/1200, 3/1200, 3.5/1200], # percentage of consolidation period high/low to breakout
    #'volume_lookback_period': [8, 10, 12, 15],
    #'volume_breakout_threshold': [1.5, 2.0, 2.5, 3.0],
    
    #'min_holding_period': [3, 4, 5, 6, 8],
    #'exit_ma_fast': [7, 9], # fast MA period
    #'exit_ma_slow': [25, 30], # slow MA period
    #'sl_ratio': [0.0005,0.0008, 0.001, 0.0015], # ratio of entry price
    #'tp_ratio': [0.03, 0.035, 0.04], # ratio of entry price
    #'trailing_stop': [False, True],
    
    #'sl_ratio': [0.005, 0.01, 0.015],
    #'tp_ratio': [0.045, 0.05],
    #'trailing_stop': [False, True]
}


pivot_params = {
    'filter': 'trade_direction',
    'row': 'adx',
    'val': 'pnlcomm',
    'func': ['COUNT', 'SUM', 'AVERAGE']
}

#pivot_params = { # ignore this if excel_pivot is False
#    'filter': 'type', # filter field
#    'filter_value': 'EXIT_POSITION', # filter value
#    'row': "prev_day_CO", # row field
#    'row_grouping': (-10, 10, 5),
#    'col': "day_so_far_CO", # column field
#    'col_grouping': (-4, 4, 2),
#   'val': "pnlcomm", # value field
#    'func': "SUM" # aggregation function
#}

# Setup data feed
datafile = 'data/VN30F1M_m1_2020-02-10_to_2025-02-14.csv'
fromdate = date(2020, 1, 1)
todate = date(2020, 12, 31)
#todate = date(2023, 2, 1) # for quick testing
data_feed = BacktestData(historical=True, hist_datafile=datafile, fromdate=fromdate, todate=todate, timeframe=bt.TimeFrame.Minutes, compression=1)

initial_cash = 5_000_000_000
#########################################
# Main function
#########################################

def main():
    optimization_results, best_params = optimize_strategy(TestStrategy, data_feed, param_grid, initial_cash)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M')}] Running with best parameters found: {best_params}")
    strategy, cerebro = run_single_backtest(TestStrategy, data_feed, initial_cash, best_params)

    performance = analyze_results(strategy)

    # Plot results if requested
    #cerebro.plot(style='candlestick')
if __name__ == '__main__':
    main()
