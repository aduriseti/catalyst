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
# targets = ['btc','eth','ltc',"etc","iota"]
# targets = ['btc','eth','ltc','dash','xmr']
# targets = ['btc','eth','ltc']
targets = ['btc','eth']
# exchangenm = "poloniex"
exchangenm = "binance"
datafreq = "daily" # | "minute"

# Bitcoin data is available from 2015-3-2. Dates vary for other tokens.
# start = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2017, 8, 16, 0, 0, 0, 0, pytz.utc)
# Bitcoin data is available from 2015-3-2. Dates vary for other tokens.
# start = datetime(2017, 1, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2019, 1, 1, 0, 0, 0, 0, pytz.utc)
# start = datetime(2019, 1, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2019, 8, 1, 0, 0, 0, 0, pytz.utc)
start = datetime(2017, 9, 1, 0, 0, 0, 0, pytz.utc)
end = datetime(2019, 9, 1, 0, 0, 0, 0, pytz.utc)

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
  N = len(targets)+1
  assetnms = [f"{ass}_{reference}" for ass in targets]
  print("Assets to balance",assetnms)
  assets = symbols(*assetnms)
  context.assets = assets
  context.nassets = N

  # rebalance interval (in days)
  context.rebint = 1
  # return interval (in days) used to compute relative returns
  context.retint = 1
  # day index - let initialization day be -1
  context.i = -1
  # initial portoflio
  # record("p",np.ones(len(N))/N)
  context.p = np.ones((1,N))/N
  # initial cummulative portfolio gradient
  # record("b",np.zeros(len(N)))
  context.b = np.zeros((1,N))
  # initial asset return - populate later
  context.r = np.ones((1,N))

  # slippage
  # context.set_slippage(spread=0.0000001)
  context.set_slippage(slippage=0.001)

verbose=False
_ctx = None
_data = None
_positions = None
_prices = None
_currentprices = None
# TODO:
# 1) support for untradable/unavailable assets
# 2) support for assets on different exchanges
# 3) plots
#   - performance
#   - allocation
#   - performance vs allocation
# 4) live execution
#   - use historical data to avoid cold start
#   - figure out how to use catalyst w/ real money
#   - simulated -> live
#   - track performance w/ persistence
# 5) risk analysis
#   - To finish the comparison between our portfolios and DJIA, lets do a very simple risk analysis with just empirical quantiles using historical results
#   - For a proper risk analysis we should model the conditional volatility of the portfolio, but this will be discussed in another post.
#   - https://insightr.wordpress.com/2017/06/22/online-portfolio-allocation-with-a-very-simple-algorithm/
# 5) support for shorting, incorporate into simplex constraints
# 6) soft window
# 7) penalize orders w/ transaction cost in ogd
# 8) minute execution
# 9) hyperparameter optimization
#   - Idea: train test split becomes temporal
#   - k-fold cross eval
#     - k-2 training folds
#     - 1 testing fold
#     - 1 holdout fold
#   - Note: k >= 3, but as k increases data available for testing/holdout lessens
#     - should hot start data be available during testing/holdout?
#   - online hyperparamter optimization - or freeze and load?
#     - freeze and load to start
# 10) volatility / volume / variance dependent hyperparameters?
# 11) concurrent arbitrage
#   - treat arbitrage as meta asset and sandbox portfolio for it w/ ogd simplex method
#     - calculate arbitrage returns
#       - pessimistic w/ max slippage
#       - from order filled amouts in context.blotter
#   - use ogd simplex method to allocate funds between 
#     - different arbitrage gap amounts - and the choice not to arbitrage
#     - arbitrage pairs
#   - 2-arbitrage -> 3-arbitrage (triangular arbitrage) -> k-arbitrage
#   - arbitrage within positions (i.e. not w/ sanboxed funds)
#     - this would involve creating meta assets per symbol
#     - arbitrage would be performed on these assets - return would be asset return + arbitrage return
# 13) incorporate prediction
#   - How?
# 14) ONS: online newton step
#   - http://www.cs.princeton.edu/~ehazan/papers/icml.pdf
# 15) Parametric policies: 
#   - https://insightr.wordpress.com/2018/02/12/parametric-portfolio-policies/
# 16) Adaptive allocation survey
#   - quantiopian starting point: https://www.quantopian.com/posts/adaptive-asset-allocation-algorithms
#   - https://systematicinvestor.wordpress.com
#   - https://cssanalytics.wordpress.com/2012/09/21/minimum-correlation-algorithm-paper-release/
#   - google search: https://www.google.com/search?q=portfolio+allocation+algorithm
def handle_data(ctx, data):
  # Only rebalance at the beginning of the algorithm execution and
  # every multiple of the rebalance period
  global _ctx,_data,_positions,_prices,_currentprices
  _ctx = ctx
  _data = data
  _positions = _positions

  currentpricedf = data.history(ctx.assets, fields='price',
                                bar_count=1, frequency='1d')
  currentprices = np.stack([currentpricedf[a].values for a in ctx.assets],
                           axis=1)
  _currentprices = currentprices
  # print("Data date:",data.current_dt)
  record(**{a.symbol:p for a,p in zip(ctx.assets,currentprices[0,:])})
  ctx.i += 1

  # if day not a multiple of return interval do nothing
  if ctx.i % ctx.rebint: return 
  
  N = ctx.nassets
  T = ctx.retint
  
  if verbose:
    print(f"Rebalance on day {ctx.i} w/ interval {T}")
  pricedf = data.history(ctx.assets, fields='price',
                         bar_count=T+1, frequency='1d')
  prices = np.stack([pricedf[a].values for a in ctx.assets],
                    axis=1)
  _prices = prices

  def concat(M,v):
    # print(v.shape,M.shape)
    M = np.concatenate((M,v.reshape(1,-1)),axis=0)
    # print(v.shape,M.shape)
    return M 
  pricegeodelta = prices[-1,:] / prices[0,:]
  # add in the option to hold usdt
  pricegeodelta = np.concatenate((pricegeodelta,np.ones((1,))))
  pricegeodelta = pricegeodelta.reshape((1,N))
  ri = pricegeodelta / np.max(pricegeodelta)
  ctx.r = concat(ctx.r,ri)
  
  p = ctx.p
  b = ctx.b
  r = ctx.r  

  bdelta =  - r[-1] / np.inner(r[-1],p[-1])
  # calculate cummulative gradient
  # bi = b[-1] + bdelta
  # bi = np.sum((*b, bdelta),axis=0)
  # print("bcum",np.round(bi,2))
  # calculate windowed gradient
  # bi = np.sum((*b[-7:], bdelta),axis=0) # 0.97
  # bi = np.sum((*b[-30:], bdelta),axis=0) # 1.06
  bi = np.sum((*b[-60:], bdelta),axis=0)
  # bi = np.sum((*b[-90:], bdelta),axis=0) # 1.17
  # bi = np.sum((*b[-365:], bdelta),axis=0) # 0.82
  # print("bwin",np.round(bi,2))
  # calculate forgetful gradient
  # bi = bdelta
  # print("bcur",np.round(bi,2))
  ctx.b = concat(ctx.b,bdelta)
  # perform one it of OGD
  # eta = 1e-4 / T
  eta = 1e-1 / T
  punc = p[-1] - eta*bi

  import cvxopt
  eps = 1e-3
  def unitsimplexprojection(v):
    d = len(v)
    P = np.eye(d)
    # q = np.zeros(d)
    q = -v
    G = np.concatenate((-np.eye(d),np.eye(d)),axis=0)
    h = np.concatenate((np.zeros(d)-eps,np.ones(d)*0.5),axis=0)
    # G = -np.eye(d)
    # h = np.zeros(d)-eps
    A = np.ones((1,d))
    b = np.array([1.])
    npMs = [P,q,G,h,A,b]
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*[cvxopt.matrix(npM) for npM in npMs])
    # print(sol["status"])
    # print(sol["x"])
    return np.array(sol["x"]).reshape((d,1))

    # SOURCE: https://scaron.info/blog/quadratic-programming-in-python.html
    # def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    #   P = .5 * (P + P.T)  # make sure P is symmetric
    #   args = [cvxopt.matrix(P), cvxopt.matrix(q)]
    #   if G is not None:
    #       args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
    #       if A is not None:
    #           args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    #   sol = cvxopt.solvers.qp(*args)
    #   if 'optimal' not in sol['status']:
    #       return None
    #   return np.array(sol['x']).reshape((P.shape[1],))

  pi = unitsimplexprojection(punc.reshape(-1))
  ctx.p = concat(ctx.p,pi)

  p = ctx.p
  b = ctx.b
  r = ctx.r

  if verbose:
    # print('r[i]',np.round(r[-1],2))
    # print("b[i]",np.round(b[-1],2))
    # print('punc[i]',np.round(punc,2))
    # print('p[i]',np.round(p[-1],2))
    print(list(zip([a.symbol for a in ctx.assets]+["usdt"],np.round(pi,3))))
    C = ctx.portfolio.cash
    V = ctx.portfolio.portfolio_value
    print("cash",C,V,C/V)
    # print("portfolio value",ctx.po)

  # order optimal weights for each asset
  for asset,percent in zip(ctx.assets,pi):
    if data.can_trade(asset):
      order_target_percent(asset,percent)

  # catalyst.api
  # figure out what I need for analysis later
  # record(pr=pr,
  #       r=r,
  #       m=m,
  #       stds=stds,
  #       max_sharpe_port=max_sharpe_port,
  #       corr_m=corr_m)

_ctx = None
_results = None
def analyze(context=None, results=None):
  global _ctx,_results
  _context,_results = context,results
  # Form DataFrame with selected data
  asyms = [a.symbol for a in context.assets]
  data = results[['portfolio_value']+asyms]
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