library(quantmod)
 
symbols = c('MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'DD', 'XOM', 'GE',
  'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE', 'PG',
  'TRV', 'UTX', 'UNH', 'VZ', 'V', 'WMT')
 
for (i in 1:length(symbols)) getSymbols(symbols[i], from = '2007-01-03', to = '2017-06-21')
 
# Building weekly returns for each of the stocks
data = sapply(symbols, function(x) Ad(to.weekly(get(x))))
data = Reduce(cbind, data)
data_returns = apply(data, 2, function(x) diff(log(x))) #log returns
colnames(data_returns)= symbols
data_returns[is.na(data_returns)] = 0 # VISA hasnt negotiations between 2007 and 2008



library(quadprogXT)
 
OGD = function(base, eta) {
 
  # Gradient of Regret Function
  gradient = function(b, p, r) b + r/(p%*%r)
 
  # Projection onto viable Set
  proj = function(p) {
 
    Dmat = diag(length(p))
    Amat = cbind(diag(rep(1, length(p))), -1)
    bvec = c(rep(0, length(p)), -1)
 
    fit = solveQPXT(Dmat = Dmat, dvec = p, Amat = Amat, bvec = bvec)
 
    return(fit$solution)
  }
 
  T = nrow(base)
  N = ncol(base)
 
  r = as.matrix(base) + 1 # this is because the algo doesnt work directly with log returns
  p = matrix(0, nrow = N, ncol = T); p[,1] = 1/N # initial portfolio
  b = matrix(0, nrow = N, ncol = T); b[,1] = 0
 
  for (i in 2:T) {
    b[,i] = gradient(b[,i-1], p[,i-1], r[i-1,]) # calculating gradient
    p.aux = p[,i-1] + eta*b[,i] # what we would like to play
    p[,i] = proj(p.aux) # projection in the viable set
  }
 
  return(list('portfolio' = p,'gradient' = b))
}
 
# testing two etas
portfolio1 = OGD(base = data_returns, eta = 1/100)
portfolio2 = OGD(base = data_returns, eta = 1/1000)


compound_return = function(portfolio, r) {
 
  return_OGD = c(); return_OGD[1] = portfolio$portfolio[,1]%*%r[1,]
  portfolio_value = c(); portfolio_value[1] = 1 + portfolio$portfolio[,1]%*%r[1,]
 
  for (i in 2:nrow(r)) {
    return_OGD[i] = portfolio$portfolio[,i]%*%r[i,]
    portfolio_value[i] = portfolio_value[i-1]*(1 + return_OGD[i])
  }  
 
 return(list('value' = portfolio_value, 'return' = return_OGD))
}

# Dow Jones
getSymbols('^DJI', src = 'yahoo', from = '2007-01-03', to = '2017-06-21')

## [1] "DJI"

DJIA_returns = as.numeric(cumprod(weeklyReturn(DJI) + 1))
 
# Individual stocks
stocks_returns = apply(data_returns + 1, 2, cumprod)
 
# Our portfolios
portfolio_value1 = compound_return(portfolio1, data_returns)
portfolio_value2 = compound_return(portfolio2, data_returns)


risk_analysis = function(return) {
 
  report = matrix(0, ncol = 2, nrow = 2)
 
  report[1,] = quantile(return, probs = c(0.01, 0.05))
  report[2,] = c(mean(return[which(return <= report[1,1])]), mean(return[which(return <= report[1,2])]))
 
  rownames(report) = c('VaR', 'CVaR')
  colnames(report) = c('1%', '5%')
 
  return(round(report, 3))
}
 
report1 = risk_analysis(portfolio_value1$return)
report2 = risk_analysis(portfolio_value2$return)
report_DJIA = risk_analysis(weeklyReturn(DJI))