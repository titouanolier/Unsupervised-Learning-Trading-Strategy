# Unsupervised-Learning-Trading-Strategy
Applies unsupervised learning (K-Means clustering) with technical indicators and portfolio optimization to build and backtest an S&amp;P 500 strategy.


This project implements a machine learning–driven portfolio strategy on S&P 500 stocks.
It:

Downloads and cleans historical stock and factor data (Fama-French, Yahoo Finance).

Computes technical indicators (RSI, Bollinger Bands, ATR, MACD, Garman-Klass volatility).

Uses K-Means clustering to group stocks with similar features.

Selects assets from the most favorable cluster each month.

Applies Efficient Frontier optimization (max Sharpe ratio) to determine portfolio weights.

Backtests the strategy against the S&P 500 (SPY), reporting cumulative returns, volatility, and Sharpe ratio.
