library(forecast)
library(knitr)

#load the data frame
issues.csv = read.csv('issues/apache_spark.csv')
issues.csv$date = as.POSIXlt(as.Date(issues.csv$date,format='%m/%d/%Y'))

to_date <- issues.csv$date[length(issues.csv$date)]
from_date <- to_date
from_date$year <- from_date$year - 1

issues.csv <- subset(issues.csv, date <= to_date & date >= from_date)
str(issues.csv)

#loading issues into a ts object
issues.ts <- ts(issues.csv$number_of_issues, frequency = 7)

time <- time(issues.ts)

n.valid <- 21
n.train <- length(issues.ts) - n.valid

train.issues.ts <- window(issues.ts, start=time[1], end=time[n.train])
valid.issues.ts <- window(issues.ts, start=time[n.train+1], end=time[n.train+n.valid])

plot(train.issues.ts, main = 'Apache Spark', bty = 'l', ylab = 'Number of Issues', xlab = 'Week',
     xlim = c(0, 54))
lines(valid.issues.ts, lwd = 2, col = 'blue')

# check whether the series is random walk
diff.ts = diff(issues.ts, lag = 1)
Acf(diff.ts, lag.max = 7, main = 'Acf on lag-1 difference')


# Naive Forecast
## Naive
train.issues.naive.pred <- naive(train.issues.ts, h = n.valid, level = 0)
plot(train.issues.naive.pred, main = 'Spark (Naive Forecast)', bty = 'l',
     ylab = 'Number of Issues')
lines(train.issues.naive.pred$fitted, lwd = 2, col = 'blue')
lines(valid.issues.ts)

plot(train.issues.ts - train.issues.naive.pred$fitted, main = 'Naive Forecast Errors Plot',
     bty = 'l', xlab = 'Week', ylab = 'Errors', xlim = c(0, 54))
lines(valid.issues.ts - train.issues.naive.pred$mean, lwd = 2, col = 'blue')
kable(accuracy(train.issues.naive.pred, valid.issues.ts))
hist(valid.issues.ts - train.issues.naive.pred$mean)

## Seasonal Naive
train.issues.snaive.pred <- snaive(train.issues.ts, h = n.valid, level = 0)
plot(train.issues.snaive.pred, main = 'Spark (Seasonal Naive Forecast)', bty = 'l',
     ylab = 'Number of Issues')
lines(train.issues.snaive.pred$fitted, lwd = 2, col = 'blue')
lines(valid.issues.ts)

plot(train.issues.ts - train.issues.snaive.pred$fitted,
     main = 'Seasonal Naive Forecast Errors Plot', bty = 'l', xlab = 'Week', ylab = 'Errors',
     xlim = c(0, 54))
lines(valid.issues.ts - train.issues.snaive.pred$mean, lwd = 2, col = 'blue')
kable(accuracy(train.issues.snaive.pred, valid.issues.ts))
hist(valid.issues.ts - train.issues.snaive.pred$mean)


# Smoothing
## Deseasonalize series
diff7 = diff(issues.ts, lag = 7)
nValid.ma = 21
nTrain.ma = length(diff7) - nValid.ma
time.diff = time(diff7)
train.ts.ma = window(diff7, start = time.diff[1], end = time.diff[nTrain.ma])
valid.ts.ma = window(diff7, start = time.diff[nTrain.ma + 1],
                     end = time.diff[nTrain.ma + nValid.ma])

plot(diff7, ylab = 'Lag-7', main = 'Deseasonalized series (Spark)', bty = 'l')

## Moving Average
library(zoo)
ma.trailing = rollmean(train.ts.ma, k = 7, align = 'right')

undiff = rep(0, length(ma.trailing))
for(j in 1:length(ma.trailing)){
  undiff[j] = ma.trailing[j] + issues.ts[j + 6]
}
undiff.ts = ts(undiff, start = time.diff[1], freq = 7)

plot(issues.ts, main = 'Spark (Moving Average)', bty = 'l', ylab = 'Number of Issues')
lines(undiff.ts, lwd = 2, col = 'blue')

last.ma = tail(undiff.ts, 1)
undiff.ts.pred = ts(rep(last.ma, 21), start = time.diff[nTrain.ma + 1], freq = 7)
lines(undiff.ts.pred, lwd = 2, lty = 2, col = 'orange')

plot(train.issues.ts[14:346] - undiff.ts, main = 'Moving Average Forecast Errors',
     ylab = 'Errors', xlab = 'Week', xlim = c(0, 54), bty = 'l')
lines(valid.issues.ts - undiff.ts.pred, lwd = 2, col = 'blue')

rmse = function(act, est) { return(sqrt(mean((act - est) ^ 2))) }
rmse(act = train.issues.ts[14:346], est = undiff.ts)
rmse(act = valid.issues.ts, est = undiff.ts.pred)

mape = function(act, est) { return(mean(abs(act - est) / act * 100)) }
mape(act = train.issues.ts[14:346], est = undiff.ts)
mape(act = valid.issues.ts, est = undiff.ts.pred)

## Exponential Smoothing
ets = ets(train.issues.ts, model = 'ZZZ', restrict = FALSE, allow.multiplicative.trend = TRUE)
summary(ets)
ets.pred = forecast(ets, h = n.valid, level = 0)

plot(ets.pred, main = 'Spark (Exponential Smoothing MNM)', bty = 'l', ylab = 'Number of Issues')
lines(ets.pred$fitted, lwd = 2, col = 'blue')
lines(valid.issues.ts)

plot(train.issues.ts - ets.pred$fitted, main = 'Exponential Smoothing (MNM) Errors Plot',
     bty = 'l', xlab = 'Week', ylab = 'Errors', xlim = c(0, 54))
lines(valid.issues.ts - ets.pred$mean, lwd = 2, col = 'blue')
kable(accuracy(ets.pred, valid.issues.ts))


# Linear Regression
## additive seasonality
train.issues.lm.additive.seasonality = tslm(train.issues.ts ~ season)
train.issues.lm.additive.seasonality.pred = forecast(train.issues.lm.additive.seasonality,
                                                     h = n.valid, level = 0)
plot(train.issues.lm.additive.seasonality.pred,
     main = 'Spark (Linear Regression Additive Seasonality)', bty = 'l',
     ylab = 'Number of Issues')
lines(train.issues.lm.additive.seasonality.pred$fitted, lwd = 2, col = 'blue')
lines(valid.issues.ts)

plot(train.issues.ts - train.issues.lm.additive.seasonality.pred$fitted,
     main = 'Linear Regression Additive Seasonality Errors Plot', bty = 'l', xlab = 'Week',
     ylab = 'Errors', xlim = c(0, 54))
lines(valid.issues.ts - train.issues.lm.additive.seasonality.pred$mean, lwd = 2, col = 'blue')
kable(accuracy(train.issues.lm.additive.seasonality.pred, valid.issues.ts))
summary(train.issues.lm.additive.seasonality)

## multiplicative seasonality
train.issues.lm.multiplicative.seasonality = tslm(train.issues.ts ~ season, lambda = 0)
train.issues.lm.multiplicative.seasonality.pred = forecast(
  train.issues.lm.multiplicative.seasonality, h = n.valid, level = 0)
plot(train.issues.lm.multiplicative.seasonality.pred,
     main = 'Spark (Linear Regression Multiplicative Seasonality)', bty = 'l',
     ylab = 'Number of Issues')
lines(train.issues.lm.multiplicative.seasonality.pred$fitted, lwd = 2, col = 'blue')
lines(valid.issues.ts)

plot(train.issues.ts - train.issues.lm.multiplicative.seasonality.pred$fitted,
     main = 'Linear Regression Multiplicative Seasonality Errors Plot', bty = 'l', xlab = 'Week',
     ylab = 'Errors', xlim = c(0, 54))
lines(valid.issues.ts - train.issues.lm.multiplicative.seasonality.pred$mean, lwd = 2,
      col = 'blue')
kable(accuracy(train.issues.lm.multiplicative.seasonality.pred, valid.issues.ts))
summary(train.issues.lm.multiplicative.seasonality)


# Autocorrelation of exponential smoothing residual series
Acf(train.issues.ts - ets.pred$fitted, lag.max = 7,
    main = 'Autocorrelation on exponential smoothing residual series')


# AR(7) model from exponential smoothing residual series
train.res.arima = Arima(train.issues.ts - ets.pred$fitted, order = c(7, 0, 0))
train.res.arima.pred = forecast(train.res.arima, h = n.valid, level = 0)

plot(train.res.arima.pred, ylab = 'Residuals', xlab = 'Week', bty = 'l', flty = 2, main = '')
lines(train.res.arima.pred$fitted, lwd = 2, col = 'blue')
lines(valid.issues.ts - ets.pred$mean)

plot(issues.ts, main = 'Spark (AR(7))', bty = 'l', ylab = 'Number of Issues', xlim = c(0, 54))
lines(ets.pred$fitted + train.res.arima.pred$fitted, lwd = 2, col = 'blue')
lines(ets.pred$mean + train.res.arima.pred$mean, lwd = 2, lty = 2, col = 'blue')

plot(train.issues.ts - ets.pred$fitted, xlim = c(0, 54),
     main = 'Forecast Error: Exponential Smoothing (MNM) vs. AR(14)', bty = 'l', ylab = 'Errors')
lines(valid.issues.ts - ets.pred$mean)
lines(train.issues.ts - (ets.pred$fitted + train.res.arima.pred$fitted), col = 'orange', lwd = 1)
lines(valid.issues.ts - (ets.pred$mean + train.res.arima.pred$mean), col = 'orange', lwd = 1)

rmse(act = train.issues.ts,
     est = (train.issues.ts - (ets.pred$fitted + train.res.arima.pred$fitted)))
rmse(act = valid.issues.ts, est = valid.issues.ts - (ets.pred$mean + train.res.arima.pred$mean))

mape(act = train.issues.ts,
     est = (train.issues.ts - (ets.pred$fitted + train.res.arima.pred$fitted)))
mape(act = valid.issues.ts, est = valid.issues.ts - (ets.pred$mean + train.res.arima.pred$mean))

Acf(train.issues.ts - (ets.pred$fitted + train.res.arima.pred$fitted), lag.max = 7,
    main = 'Autocorrelations of residuals-of-residuals series')


# external info
xTrain = data.frame(isCommit = issues.csv$is_commit[1:n.train])
xTest = data.frame(isCommit = issues.csv$is_commit[(n.train + 1):(n.train + n.valid)])
stlm.reg.fit = stlm(train.issues.ts, s.window = 'periodic', xreg = xTrain, method = 'arima')
stlm.reg.fit$model
stlm.reg.pred = forecast(stlm.reg.fit, xreg = xTest, h = n.valid, level = 0)

plot(stlm.reg.pred, ylab = 'Number of Issues', xlab = 'Week', bty = 'l')
lines(stlm.reg.pred$fitted, col = 'blue', lwd = 2)
lines(valid.issues.ts)

plot(train.issues.ts - stlm.reg.pred$fitted, main = 'External Info Errors Plot', bty = 'l',
     ylab = 'Errors', xlab = 'Week', xlim = c(0, 54))
lines(valid.issues.ts - stlm.reg.pred$mean, col = 'blue', lwd = 2)
kable(accuracy(stlm.reg.pred, valid.issues.ts))
