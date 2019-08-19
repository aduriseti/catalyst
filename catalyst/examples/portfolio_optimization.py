#%%
'''Use this code to execute a portfolio optimization model. This code
   will select the portfolio with the maximum Sharpe Ratio. The parameters
   are set to use 180 days of historical data and rebalance every 30 days.

   This is the code used in the following article:
   https://blog.enigma.co/markowitz-portfolio-optimization-for-cryptocurrencies-in-catalyst-b23c38652556

   You can run this code using the Python interpreter:

   $ python portfolio_optimization.py
'''
from __future__ import division
import os
os.getcwd()
os.chdir(os.path.expanduser("~"))
import pytz
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from catalyst.api import record, symbols, symbol, order_target_percent
from catalyst.utils.run_algo import run_algorithm

import ccxt
exch = ccxt.binance()
exch.loadMarkets()
syms = exch.symbols
usdsyms = [s for s in syms if "usd" in s.lower()]
print(usdsyms)

reference = "usdt"
# targets = ['btc','eth','ltc',"etc","iota",'dash','xmr']
targets = ['btc','eth','ltc',"etc","iota"]
# targets = ['btc','eth','ltc','dash','xmr']
# targets = ['btc','eth','ltc']
# exchangenm = "poloniex"
exchangenm = "binance"
datafreq = "daily" # | "minute"

# Bitcoin data is available from 2015-3-2. Dates vary for other tokens.
# start = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2017, 8, 16, 0, 0, 0, 0, pytz.utc)
# Bitcoin data is available from 2015-3-2. Dates vary for other tokens.
# start = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2019, 1, 1, 0, 0, 0, 0, pytz.utc)
start = datetime(2018, 9, 14, 0, 0, 0, 0, pytz.utc)
end = datetime(2019, 8, 14, 0, 0, 0, 0, pytz.utc)

import catalyst
import catalyst.exchange
import catalyst.exchange.exchange
import catalyst.exchange.exchange_bundle
#%%
def initialize(context):
  # Portfolio assets list
  exbundle = catalyst.exchange.exchange_bundle.ExchangeBundle(exchangenm)
  # exbundle.clean(datafreq)
  context.exbundle = exbundle
  assetnms = [f"{ass}_{reference}" for ass in targets]
  print("Assets to balance",assetnms)
  context.assets = symbols(*assetnms)
  context.nassets = len(context.assets)

  # Set the time window that will be used to compute expected return
  # and asset correlations
  # context.window = 180
  context.window = 90
  # Set the number of days between each portfolio rebalancing
  # context.rebalance_period = 30
  context.rebalance_period = 7
  context.i = 0

# import catalyst.exchange
_context = None
_data = None
_positions = None
_prices = None
def handle_data(context, data):
  # Only rebalance at the beginning of the algorithm execution and
  # every multiple of the rebalance period
  currentprices = data.history(context.assets, fields='price',
                               bar_count=1, frequency='1d')
  global _context,_data,_positions,_prices
  _context = context
  _data = data
  _positions = _positions
  # print("Data date:",data.current_dt)
  # print(currentprices)
  record(
    **{a.symbol:currentprices[a].values[0] for a in context.assets}
  )
  if context.i == 0 or context.i % context.rebalance_period == 0:
    n = context.window
    print(f"Rebalance {context.i} w/ window size {n}")
    prices = data.history(context.assets, fields='price',
                bar_count=n + 1, frequency='1d')
    _prices = prices
    # pr = np.asmatrix(prices)
    pr = np.array(prices)
    t_prices = prices.iloc[1:n + 1]
    t_val = t_prices.values
    tminus_prices = prices.iloc[0:n]
    tminus_val = tminus_prices.values
    # Compute daily returns (r)
    # r = np.asmatrix(t_val / tminus_val - 1)
    r = np.array(t_val / tminus_val - 1)
    # Compute the expected returns of each asset with the average
    # daily return for the selected time window
    # m = np.asmatrix(np.mean(r, axis=0))
    m = np.array(np.mean(r, axis=0))
    # ###
    # Note: 
    # - include langrangian term in variation - enforces stability
    #   - possible interpretation is the assumption that all price data has some uncertainty beyound 
    #     variance calculated from observed data
    # - possibly a better approach is to cancel rebalancing if data has no variance
    # - is this interpolation performed by cataylst? - if so, a first or higher order interpolation could
    #   fix this error
    # - alternatively, could introduce noise to interpolated value
    # - should I selectively introduce extremely high variance to assets w/ no variance?
    stds = np.std(r, axis=0) + 0.01
    # original code
    # stds = np.std(r, axis=0)
    # Compute excess returns matrix (xr)
    xr = r - m
    # Matrix algebra to get variance-covariance matrix
    cov_m = np.dot(np.transpose(xr), xr) / n
    # Compute asset correlation matrix (informative only)
    corr_m = cov_m / np.dot(np.transpose(stds), stds)

    # Define portfolio optimization parameters
    n_portfolios = 50000
    results_array = np.zeros((3 + context.nassets, n_portfolios))
    for p in range(n_portfolios):
      weights = np.random.random(context.nassets)
      weights /= np.sum(weights)
      w = np.asmatrix(weights)
      p_r = np.sum(np.dot(w, np.transpose(m))) * 365
      p_std = np.sqrt(np.dot(np.dot(w, cov_m),
                   np.transpose(w))) * np.sqrt(365)

      # store results in results array
      results_array[0, p] = p_r
      results_array[1, p] = p_std
      # store Sharpe Ratio (return / volatility) - risk free rate element
      # excluded for simplicity
      results_array[2, p] = results_array[0, p] / (results_array[1, p] + 1e5)

      for i, w in enumerate(weights):
        results_array[3 + i, p] = w

    columns = ['r', 'stdev', 'sharpe'] + context.assets

    # convert results array to Pandas DataFrame
    results_frame = pd.DataFrame(np.transpose(results_array),
                   columns=columns)
    # locate position of portfolio with highest Sharpe Ratio
    # print(f"Best idx:",results_frame['sharpe'].idxmax())
    # print(f"results_frame:", results_frame.head())
    max_sharpe_port = results_frame.iloc[results_frame['sharpe'].idxmax()]
    # locate positon of portfolio with minimum standard deviation
    # min_vol_port = results_frame.iloc[results_frame['stdev'].idxmin()]

    # order optimal weights for each asset
    for asset in context.assets:
      if data.can_trade(asset):
        order_target_percent(asset, max_sharpe_port[asset])

    # # create scatter plot coloured by Sharpe Ratio
    # plt.scatter(results_frame.stdev,
    #       results_frame.r,
    #       c=results_frame.sharpe,
    #       cmap='RdYlGn')
    # plt.xlabel('Volatility')
    # plt.ylabel('Returns')
    # plt.colorbar()

    # # plot blue circle to highlight position of portfolio
    # # with highest Sharpe Ratio
    # plt.scatter(max_sharpe_port[1],
    #       max_sharpe_port[0],
    #       marker='o',
    #       color='b',
    #       s=200)

    # plt.show()
    # print(max_sharpe_port)

    record(pr=pr,
         r=r,
         m=m,
         stds=stds,
         max_sharpe_port=max_sharpe_port,
         corr_m=corr_m)

  context.i += 1

_context = None
_results = None
def analyze(context=None, results=None):
  global _context,_results
  _context,_results = context,results
  # Form DataFrame with selected data
  # data = results[['pr', 'r', 'm', 'stds', 'max_sharpe_port', 'corr_m',
  #         'portfolio_value']]
  asyms = [a.symbol for a in context.assets]
  data = results[['corr_m','portfolio_value']+asyms]
  print(pd.DataFrame.head(data,5))
  print(pd.DataFrame.tail(data,5))
  strats = asyms + ["portfolio_value"]
  print(list(zip(strats,data[strats].iloc[-1] / data[strats].iloc[0])))

  # # Save results in CSV file
  # filename = os.path.splitext(os.path.basename(__file__))[0]
  # data.to_csv(filename + '.csv')
#%%
if __name__ == '__main__':
  results = run_algorithm(initialize=initialize,
              data_frequency=datafreq,
              handle_data=handle_data,
              analyze=analyze,
              start=start,
              end=end,
              exchange_name=exchangenm,
              capital_base=100000,
              quote_currency='usdt')
#%%
asyms = [a.symbol for a in _context.assets]
data = _results[['corr_m','portfolio_value']+asyms]
print(pd.DataFrame.head(data,5))
print(pd.DataFrame.tail(data,5))
strats = asyms + ["portfolio_value"]
print(list(zip(strats,data[strats].iloc[-1] / data[strats].iloc[0])))
#%%
data[strats]
#%%
old = _data.current_dt
#%%
import datetime
#%%
_data.history(_context.assets,fields='price',
                bar_count=5, frequency='1d')
#%%
_data.__dir__()

#%%
_data.current_dt

#%%
_prices

#%%
_data.current(_context.assets,fields='price',)
#%%
_data.is_stale(_context.assets)
#%%
type(_context)


#%%
