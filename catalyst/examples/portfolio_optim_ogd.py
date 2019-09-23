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

# TODO: figure out how to filter symbol by volume and activitiy to decide
#       what universe of symbols to trade on
import ccxt
exch = ccxt.binance()
markets = exch.loadMarkets()
syms = exch.symbols
print("Exchange symbols:", syms)
usdtsyms = [s for s in syms if "usdt" == s.split("/")[1].lower()]
print("USDT symbols",usdtsyms)

reference = "usdt"
# dont know how to deal w/ inactive markets - blacklist them
blacklist = ["beam","bch"] + \
            [s for s,m in markets.items() if not m["active"]]
# targets = [s.split("/")[0].lower() for s in usdtsyms]
# targets = ['btc','eth','ltc',"etc","iota",'dash','xmr',"ada","algo","beam","xrp","zrx","bat","bnb","bch",
#            "xlm","bchabc","zec","waves","qtum","mith"]
targets = ['btc','eth','ltc',"etc","iota",'dash','xmr',"ada","algo","beam","xrp","zrx","bat","bnb","bch"]
# targets = ['btc','eth','ltc',"etc","iota",'dash','xmr']
# targets = ['btc','eth','ltc',"etc","iota"]
# targets = ['btc','eth','ltc','dash','xmr']
# targets = ['btc','eth','ltc']
# targets = ['btc','eth']
# exchangenm = "poloniex"
targets = [t for t in targets if t not in blacklist]
print("currency targets",targets)
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
# start = datetime(2017, 9, 1, 0, 0, 0, 0, pytz.utc)
# end = datetime(2017, 10, 1, 0, 0, 0, 0, pytz.utc)

import catalyst
import catalyst.exchange
import catalyst.exchange.exchange
import catalyst.exchange.exchange_bundle

import typing
import numpy as np
import scipy as sp
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
  context.assets: typing.Iterable[catalyst.api.symbol] = assets
  context.nassets = N

  # TODO: should I ever have return interval != rebalance interval?
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
  context.set_commission(maker=0.001, taker=0.002)

# TODO:
# 1) [x] support for untradable/unavailable assets
#   - fixed constraint incompatibility bug
# 2) [ ] support for assets on different exchanges
#   - not that important for the timescales I operate on - defer until incorporate arbitrage 
#   - or until I decrease timescale
# 3) [ ] newton step?
#   - not structurally/procedurally important - defer until proper evaluation framework is built
# 4) plots
#   - [x] performance
#   - [x] allocation
#   - [x] performance vs allocation
# 5) live execution
#   - use historical data to avoid cold start
#   - figure out how to use catalyst w/ real money
#   - simulated -> live
#   - deployment
#   - track performance w/ persistence
# 6) risk analysis
#   - To finish the comparison between our portfolios and DJIA, lets do a very simple risk analysis with just empirical quantiles using historical results
#   - For a proper risk analysis we should model the conditional volatility of the portfolio, but this will be discussed in another post.
#   - https://insightr.wordpress.com/2017/06/22/online-portfolio-allocation-with-a-very-simple-algorithm/
# 7) support for shorting, incorporate into simplex constraints
# 8) soft window
# 9) penalize orders w/ transaction cost in ogd
# 10) minute execution
# 11) hyperparameter optimization
#   - Idea: train test split becomes temporal
#   - k-fold cross eval
#     - k-2 training folds
#     - 1 testing fold
#     - 1 holdout fold
#   - Note: k >= 3, but as k increases data available for testing/holdout lessens
#     - should hot start data be available during testing/holdout?
#   - online hyperparamter optimization - or freeze and load?
#     - freeze and load to start
# 12) volatility / volume / variance dependent hyperparameters?
# 13) concurrent arbitrage
#   - treat arbitrage as meta asset and sandbox portfolio for it w/ ogd simplex method
#     - calculate arbitrage returns
#       - pessimistic w/ max slippage
#       - from order filled amouts in context.blotter
#   - use ogd simplex method to allocate funds between 
#     - different arbitrage gap amounts - and the choice not to arbitrage
#     - arbitrage pairs (extensible to arbitrage paths later)
#   - 2-arbitrage -> 3-arbitrage (triangular arbitrage) -> k-arbitrage
#   - arbitrage within positions (i.e. not w/ sanboxed funds)
#     - this would involve creating meta assets per symbol
#     - arbitrage would be performed on these assets - return would be asset return + arbitrage return
# 14) incorporate prediction
#   - How?
# 15) ONS: online newton step
#   - http://www.cs.princeton.edu/~ehazan/papers/icml.pdf
# 16) Parametric policies: 
#   - https://insightr.wordpress.com/2018/02/12/parametric-portfolio-policies/
# 17) Adaptive allocation survey
#   - quantiopian starting point: https://www.quantopian.com/posts/adaptive-asset-allocation-algorithms
#   - https://systematicinvestor.wordpress.com
#   - https://cssanalytics.wordpress.com/2012/09/21/minimum-correlation-algorithm-paper-release/
#   - google search: https://www.google.com/search?q=portfolio+allocation+algorithm
verbose=1
_ctx = None
_data = None
_positions = None
_prices = None
_currentprices = None
def handle_data(ctx, data):
  # Only rebalance at the beginning of the algorithm execution and
  # every multiple of the rebalance period
  global _ctx,_data,_positions,_prices,_currentprices
  _ctx = ctx
  _data = data
  _positions = _positions

  # currentpricedf = data.history(ctx.assets, fields='price',
  #                               bar_count=1, frequency='1d')
  assets: typing.Iterable[catalyst.api.symbol] = ctx.assets
  # price of -1 indicates that asset is unavailable (i.e. does not exist at this point)
  # TODO: differentiate b/w assets that haven't been ingested yet and assets that have
  #       been ingested but dont have data extending back to this point in time
  def safehist(assets: typing.Iterable[catalyst.api.symbol],barcount: int,
               default:float=-1):
    def safehistgen():
      for a in assets:
        # print(f"Getting price for asset {a.symbol}")
        try:
          asshist = data.history(a,fields='price',bar_count=barcount,frequency='1d').values
          assert len(asshist) == barcount
          yield asshist
        # catalyst.exchange.exchange_errors.NoCandlesReceivedFromExchange
        except Exception as ex:
          if verbose:
            print(f"Getting price for asset {a.symbol} failed w/ {type(ex)}:\n{ex}")
          yield np.ones((barcount,)) * default
    return np.stack([*safehistgen()],axis=1)
  
  # get price block
  N = ctx.nassets
  T = ctx.retint
  prices = safehist(assets,T+1)
  # print("barcount",T+1,"prices",prices.shape)
  # print([*zip([a.symbol for a in assets],prices)])
  _prices = prices
  
  record(**{a.symbol:p for a,p in zip(ctx.assets,prices[-1,:])})
  ctx.i += 1

  # if day not a multiple of return interval do nothing
  if ctx.i % ctx.rebint: return 
  
  if verbose:
    print(f"Rebalance on day {ctx.i} w/ interval {T}")

  def concat(M,v):
    # print(v.shape,M.shape)
    M = np.concatenate((M,v.reshape(1,-1)),axis=0)
    # print(v.shape,M.shape)
    return M 

  # assets for which prices are available (and > 0) every day in the return interval 
  #   are considered to be available for trading
  # additionally use the data context to flag assets as avail / not avail
  availidx = ~((prices <= 0).sum(axis=0) > 0)
  # availidx = availidx & np.array([data.can_trade(a) for a in assets])
  pricegeodelta = prices[-1,:] / prices[0,:]
  # optimistic: consider unavailbale assets to have the best return in a period
  # pricegeodelta[~availidx] = max(pricegeodelta[availidx].max(),1) # add in option to hold usdt/cash
  # nuetral: consider unavailable assets to have no positive or negative return
  # pricegeodelta[~availidx] = 1
  # pessimistic: consider unavailble assets to have the minimal return in a period
  pricegeodelta[~availidx] = min(pricegeodelta[availidx].min(),1) # add in option to hold usdt/cash
  # add in the option to hold usdt
  availidx = np.concatenate((availidx,np.array([True])))
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
  # bi = bdelta
  # bi = np.sum((*b[-7:], bdelta),axis=0) # 0.97
  # bi = np.sum((*b[-30:], bdelta),axis=0) # 1.06
  # bi = np.sum((*b[-60:], bdelta),axis=0)
  # bi = np.sum((*b[-90:], bdelta),axis=0) # 1.17
  # bi = np.sum((*b[-180:], bdelta),axis=0) # 1.17
  # bi = np.sum((*b[-365:], bdelta),axis=0) # 0.82
  # bi = np.sum((*b, bdelta),axis=0) # 0.82
  # bi = bdelta
  # bi = np.mean((*b[-7:], bdelta),axis=0) # 0.97
  bi = np.mean((*b[-30:], bdelta),axis=0) # 1.06
  # bi = np.mean((*b[-60:], bdelta),axis=0)
  # bi = np.mean((*b[-90:], bdelta),axis=0) # 1.17
  # bi = np.mean((*b[-180:], bdelta),axis=0) # 1.17
  # bi = np.mean((*b[-365:], bdelta),axis=0) # 0.82
  # bi = np.mean((*b, bdelta),axis=0) # 0.82
  # print("bwin",np.round(bi,2))
  # calculate forgetful gradient
  # bi = bdelta
  # print("bcur",np.round(bi,2))
  ctx.b = concat(ctx.b,bdelta)
  # perform one it of OGD
  # scale ogd step by rebal interval, what about return interval?
  # eta = 1e-1 * ctx.rebint
  # eta = 1e0 * ctx.rebint
  eta = 3e-1 * ctx.rebint
  punc = p[-1] - eta*bi

  import cvxopt
  # support additional constraints
  def psuedounitsimplexprojection(v,Gadd=None,hadd=None,Aadd=None,badd=None,eps=1e-4):
    d = len(v)

    # additional constraints empty by defualt
    # Am I missing something here? - dont think this makes the constraints empty
    # G = G if G is not None else np.ones((0,))
    # h = h if h is not None else np.ones((0,d))
    # A = A if A is not None else np.ones((0,d))
    # b = b if b is not None else np.ones((0,))

    # w.t. find argmin(x) [||x-v|| == (x.T@x - 2x.T@v)]
    # min 1/2*x.T@P@x + q.T@x
    P = np.eye(d)
    q = -v
    # s.t.
    # add in new constraints
    # TODO: detect/resolve contradictory constraints
    G = -np.eye(d)
    if Gadd is not None: G = np.concatenate((G,Gadd),axis=0)
    # print("|G|",G.shape)
    # print(G)
    # by default, require bot to hold at least eps of each currency - helps w/ stability
    # Note: this means default simplex projections is not really a simplex
    h = np.zeros(d)-eps
    if hadd is not None: h = np.concatenate((h,hadd),axis=0)
    # print("|h|",h.shape)
    # print(h)
    # total portfolio value proportions must be 1
    A = np.ones((1,d)) 
    if Aadd is not None: A = np.concatenate((A,Aadd),axis=0)
    # print("|A|",A.shape)
    # print(A)
    b = np.array([1.])
    if badd is not None: b = np.concatenate((b,badd),axis=0)
    # print("|b|",b.shape)
    # print(b)
    
    npMs = [P,q,G,h,A,b]
    cvxopt.solvers.options['show_progress'] = False
    sol = cvxopt.solvers.qp(*[cvxopt.matrix(npM) for npM in npMs])
    # print(sol["status"],sol["x"])
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
  def unitsimplexprojection(v,eps=1e-4):
    d = len(v)
    P = np.eye(d)
    q = -v
    G = -np.eye(d)
    # h = np.zeros(d)-eps
    h = np.zeros(d)
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

  eps = 1e-6
  d = np.prod(punc.shape)
  # constrain maximum amt of portfolio put into 1 asset
  G = np.eye(d)
  h = np.ones((d,))*0.9
  # additionally, prevent us from holding unavailable assets
  A = np.eye(d)[~availidx]
  # Note: because we force the bot to allocate at least eps to each asset
  #       we cannot enforce that unavailable assets will have 0 allocation
  #       so instead set them to eps
  #   - could also set eps to 0
  b = np.ones((d,))[~availidx] * eps
  try: 
    pi = psuedounitsimplexprojection(punc.reshape(-1),G,h,A,b,eps)
  except Exception as ex:
    print(ex)
    print("unconstrained portfolio:",punc)
    return
  ctx.p = concat(ctx.p,pi)

  p = ctx.p
  b = ctx.b
  r = ctx.r

  # print(list(zip([a.symbol for a in ctx.assets]+["usdt"],np.round(pi,3))))
  if verbose:
    # print('r[i]',np.round(r[-1],2))
    # print("b[i]",np.round(b[-1],2))
    # print('punc[i]',np.round(punc,2))
    # print('p[i]',np.round(p[-1],2))
    asyms = [a.symbol for a in ctx.assets]+["usdt"]
    print("Geometric returns:\n",[*zip(asyms,pricegeodelta)])
    print("Unconstrained allocations:\n",[*zip(asyms,np.round(punc,3))])
    print("Constrained allocations:\n",[*zip(asyms,np.round(pi,3))])
    C = ctx.portfolio.cash
    V = ctx.portfolio.portfolio_value
    print("cash",C,"portfolio_value",V,"proportion value in cash",C/V)

  # order optimal weights for each asset
  for asset,avail,percent in zip(assets,availidx,pi):
    if verbose: 
      print(f"Asset {asset.symbol} availability: {avail}, allocating {percent} of portfolio")
    # dont bother allocating less than 1% of portfolio
    if avail and percent >= 1e-2:
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
def analyze(ctx=None, results=None):
  global _ctx,_results
  _ctx,_results = ctx,results
  # Form DataFrame with selected data
  asyms = [a.symbol for a in ctx.assets]
  data = results[['portfolio_value']+asyms]
  if verbose:
    print(pd.DataFrame.head(data,5))
    print(pd.DataFrame.tail(data,5))
  strats = asyms + ["portfolio_value"]
  # TODO: calculate returns for coins only from when they are first available
  firstlistingidx = np.argmax(data[strats].values > 0,axis=0)
  print(firstlistingidx)
  print([*zip(asyms,firstlistingidx)])
  # returns = data[strats] / data[strats].iloc[0]
  returns = data[strats] / np.diag(data[strats].values[firstlistingidx])
  # print(list(zip(strats,np.round(returns,3))))
  print(list(zip(strats,returns.iloc[-1])))
  
  LO = plt.stackplot(range(ctx.p.shape[0]),*ctx.p.T)
  plt.legend(iter(LO),asyms+["usdt"])
  plt.show()

  portfolio_returns = returns["portfolio_value"].values.reshape(-1,1)[1:]
  portfolio_returns = np.array([el for R in portfolio_returns for el in R])
  M = _ctx.p[-portfolio_returns.shape[0]:,:] * portfolio_returns
  LO = plt.stackplot(range(M.shape[0]),*M.T,baseline="zero")
  plt.legend(iter(LO),asyms+["usdt"])
  plt.show()

  # plt.plot(results["portfolio_value"])
  # plt.show()
  LO = plt.plot(returns)
  plt.legend(iter(LO),strats)
  plt.show()

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
# asyms = [a.symbol for a in _ctx.assets]
# data = results[['portfolio_value']+asyms]
# if verbose:
#   print(pd.DataFrame.head(data,5))
#   print(pd.DataFrame.tail(data,5))
# strats = asyms + ["portfolio_value"]
# # TODO: calculate returns for coins only from when they are first available
# #%%
# data[strats].values
# #%%
# firstlistingidx = np.argmax(data[strats].values > 0, axis=0)
# firstlistingidx
# #%%
# firstlistingidx.shape
# #%%
# print(firstlistingidx)
# print([*zip(asyms,firstlistingidx)])
# #%%
# # returns = data[strats] / data[strats].iloc[0]
# returns = data[strats] / np.diag(data[strats].values[firstlistingidx])
# # print(list(zip(strats,np.round(returns,3))))
# print(list(zip(strats,returns.iloc[-1])))
#%%
