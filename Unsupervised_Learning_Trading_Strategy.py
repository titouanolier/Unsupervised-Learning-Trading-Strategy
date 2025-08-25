from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import pandas_ta
import warnings
warnings.filterwarnings('ignore')

# df_stocks = pd.read_csv("2025_sp_500_stocks.csv")
# tickers = df_stocks["Symbol"].dropna().unique().tolist()

sp500 = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')

symbols_list = sp500['Symbol'].unique().tolist()

end_date = '2025-08-20'

start_date = pd.to_datetime(end_date)-pd.DateOffset(365*8)

df = yf.download(tickers=symbols_list,
                 start=start_date,
                 end=end_date).stack()

df.index.names = ['date', 'ticker']

df.columns = df.columns.str.lower()

#print(df)

# 2. Calculate features and technical indicators for each stock.


df['garman_klass_vol'] = ((np.log(df['high']) - np.log(df['low'])) ** 2) / 2 - (2 * np.log(2) - 1) * (
            (np.log(df['close']) - np.log(df['open'])) ** 2)


df['rsi'] = df.groupby(level=1)['close'].transform(lambda x: pandas_ta.rsi(close=x, length=20))

df['bb_low'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 0])

df['bb_mid'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 1])

df['bb_high'] = df.groupby(level=1)['close'].transform(
    lambda x: pandas_ta.bbands(close=np.log1p(x), length=20).iloc[:, 2])


def compute_atr(stock_data):
    atr = pandas_ta.atr(high=stock_data['high'],
                        low=stock_data['low'],
                        close=stock_data['close'],
                        length=14)
    return atr.sub(atr.mean()).div(atr.std())


df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)


def compute_macd(close):
    macd = pandas_ta.macd(close=close, length=20).iloc[:, 0]
    return macd.sub(macd.mean()).div(macd.std())


df['macd'] = df.groupby(level=1, group_keys=False)['close'].apply(compute_macd)

df['dollar_volume'] = (df['close'] * df['volume']) / 1e6



#Aggregate to monthly level and filter top 150 most liquid stock for each month.

last_cols = [c for c in df.columns.unique(0) if c not in ['dollar_volume', 'volume', 'open',
                                                          'high', 'low', 'close']]

data = (pd.concat([df.unstack('ticker')['dollar_volume'].resample('M').mean().stack('ticker').to_frame('dollar_volume'),
                   df.unstack()[last_cols].resample('M').last().stack('ticker')],
                  axis=1)).dropna()

# Calculate 5-year rolling average of dollar volume for each stocks before filtering.

data['dollar_volume'] = (data.loc[:, 'dollar_volume'].unstack('ticker').rolling(5*12, min_periods=12).mean().stack())

data['dollar_vol_rank'] = (data.groupby('date')['dollar_volume'].rank(ascending=False))

data = data[data['dollar_vol_rank']<150].drop(['dollar_volume', 'dollar_vol_rank'], axis=1)


#Calculate Monthly Returns for different time horizons as features
#
# print(df.columns)


def calculate_returns(df):
    outlier_cutoff = 0.005

    lags = [1, 2, 3, 6, 9, 12]

    for lag in lags:
        df[f'return_{lag}m'] = (df['close']
                                .pct_change(lag)
                                .pipe(lambda x: x.clip(lower=x.quantile(outlier_cutoff),
                                                       upper=x.quantile(1 - outlier_cutoff)))
                                .add(1)
                                .pow(1 / lag)
                                .sub(1))
    return df

data['close'] = df['close']

data = data.groupby(level=1, group_keys=False).apply(calculate_returns).dropna()

# print(data)
# print(data.columns)

# 5. Download Fama-French Factors and Calculate Rolling Factor Betas.


factor_data = web.DataReader('F-F_Research_Data_5_Factors_2x3',
                               'famafrench',
                               start='2010')[0].drop('RF', axis=1)

factor_data.index = factor_data.index.to_timestamp()

factor_data = factor_data.resample('M').last().div(100)

factor_data.index.name = 'date'

factor_data = factor_data.join(data['return_1m']).sort_index()

#print(factor_data)

#Filter out stocks with less than 10 months of data.gg

observations = factor_data.groupby(level=1).size()

valid_stocks = observations[observations >= 10]

factor_data = factor_data[factor_data.index.get_level_values('ticker').isin(valid_stocks.index)]

#print(factor_data)

#Calculate Rolling Factor Betas.

betas = (factor_data.groupby(level=1,
                            group_keys=False)
         .apply(lambda x: RollingOLS(endog=x['return_1m'],
                                     exog=sm.add_constant(x.drop('return_1m', axis=1)),
                                     window=min(24, x.shape[0]),
                                     min_nobs=len(x.columns)+1)
         .fit(params_only=True)
         .params
         .drop('const', axis=1)))

# print(betas)

# Join the rolling factors data to the main features dataframe.

factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']

data = (data.join(betas.groupby('ticker').shift()))

data.loc[:, factors] = data.groupby('ticker', group_keys=False)[factors].apply(lambda x: x.fillna(x.mean()))

data = data.drop('close', axis=1)

data = data.dropna()

# print(data.info())

#DATA is ready now
# For each month fit a K-Means Clustering Algorithm to group similar assets based on their features.
# let's identify the clusters

from sklearn.cluster import KMeans


# Apply pre-defined centroids in order to get the most high RSI value at the top..

target_rsi_values = [30, 45, 55, 70]

initial_centroids = np.zeros((len(target_rsi_values), 18))

initial_centroids[:, 1] = target_rsi_values

# print(initial_centroids)

def get_clusters(df):
    df['cluster'] = KMeans(n_clusters=4,
                           random_state=0,
                           init=initial_centroids).fit(df).labels_
    return df

data = data.dropna().groupby('date', group_keys=False).apply(get_clusters)

# print(data)
# print(data.columns)

#data = data.drop('cluster', axis=1)

#print(data.columns)
#
#
def plot_clusters(data):
    cluster_0 = data[data['cluster'] == 0]
    cluster_1 = data[data['cluster'] == 1]
    cluster_2 = data[data['cluster'] == 2]
    cluster_3 = data[data['cluster'] == 3]

    plt.scatter(cluster_0.iloc[:, 5], cluster_0.iloc[:, 1], color='red', label='cluster 0')
    plt.scatter(cluster_1.iloc[:, 5], cluster_1.iloc[:, 1], color='green', label='cluster 1')
    plt.scatter(cluster_2.iloc[:, 5], cluster_2.iloc[:, 1], color='blue', label='cluster 2')
    plt.scatter(cluster_3.iloc[:, 5], cluster_3.iloc[:, 1], color='black', label='cluster 3')

    plt.legend()
    plt.show()
    return


plt.style.use('ggplot')

for i in data.index.get_level_values('date').unique().tolist():
    g = data.xs(i, level=0)



# For each month select assets based on the cluster and form a portfolio
# based on Efficient Frontier max sharpe ratio optimization

filtered_df = data[data['cluster'] == 3].copy()

filtered_df = filtered_df.reset_index(level=1)

filtered_df.index = filtered_df.index + pd.DateOffset(1)

filtered_df = filtered_df.reset_index().set_index(['date', 'ticker'])

dates = filtered_df.index.get_level_values('date').unique().tolist()

fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()


#Define portfolio optimization function


from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns


def optimize_weights(prices, lower_bound=0):
    returns = expected_returns.mean_historical_return(prices=prices,
                                                      frequency=252)

    cov = risk_models.sample_cov(prices=prices,
                                 frequency=252)

    ef = EfficientFrontier(expected_returns=returns,
                           cov_matrix=cov,
                           weight_bounds=(lower_bound, .1),
                           solver='SCS')

    weights = ef.max_sharpe()

    return ef.clean_weights()

stocks = data.index.get_level_values('ticker').unique().tolist()

new_df = yf.download(tickers=stocks,
                     start=data.index.get_level_values('date').unique()[0]-pd.DateOffset(months=12),
                     end=data.index.get_level_values('date').unique()[-1])


#Calculate daily returns for each stock which could land up in our portfolio.
#Then loop over each month start, select the stocks for the month and calculate their weights for the next month.
#If the maximum sharpe ratio optimization fails for a given month, apply equally-weighted weights
#Calculated each day portfolio return.

returns_dataframe = np.log(new_df['Close']).diff()

# 2. DataFrame final
portfolio_df = pd.DataFrame()

for start_date in fixed_dates.keys():
    try:
        # Définir la fin du mois
        end_date = (pd.to_datetime(start_date) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

        # Actions sélectionnées
        cols = fixed_dates[start_date]

        # Fenêtre d'optimisation = 12 mois avant start_date
        optimization_start_date = (pd.to_datetime(start_date) - pd.DateOffset(months=12)).strftime('%Y-%m-%d')
        optimization_end_date = (pd.to_datetime(start_date) - pd.DateOffset(days=1)).strftime('%Y-%m-%d')

        optimization_df = new_df['Close'].loc[optimization_start_date:optimization_end_date, cols]

        # -------------------
        # Étape 1 : Optimisation ou fallback en equally-weighted
        # -------------------
        try:
            weights = optimize_weights(prices=optimization_df,
                                       lower_bound=round(1 / (len(optimization_df.columns) * 2), 3))
            weights = pd.DataFrame.from_dict(weights, orient="index", columns=["weight"])
        except:
            print(f"Max Sharpe Optimization failed for {start_date}, Continuing with Equal-Weights")
            weights = pd.DataFrame(1 / len(optimization_df.columns),
                                   index=optimization_df.columns,
                                   columns=["weight"])

        # -------------------
        # Étape 2 : Récupérer les rendements du mois en cours
        # -------------------
        temp_df = returns_dataframe.loc[start_date:end_date, cols]

        temp_df = temp_df.stack().to_frame('return').reset_index()  # ["Date", "Ticker", "return"]

        # -------------------
        # Étape 3 : Merger avec les poids
        # -------------------
        temp_df = temp_df.merge(weights, left_on="Ticker", right_index=True, how="left")

        # -------------------
        # Étape 4 : Calculer le rendement pondéré
        # -------------------
        temp_df["weighted_return"] = temp_df["return"] * temp_df["weight"]

        # Somme par jour
        temp_df = temp_df.groupby("Date")["weighted_return"].sum().to_frame("Strategy Return")

        # Concat avec le portefeuille global
        portfolio_df = pd.concat([portfolio_df, temp_df])

    except Exception as e:
        print(e)

# Nettoyage final
portfolio_df = portfolio_df[~portfolio_df.index.duplicated(keep="first")]

print(portfolio_df)


#Visualize Portfolio returns and compare to SP500 returns.

spy = yf.download(tickers='SPY',
                  start='2015-01-01',
                  end=dt.date.today())

# Calcul des rendements SPY
spy_ret = np.log(spy[['Close']]).diff().dropna().rename({'Close':'SPY Buy&Hold'}, axis=1)

# Aplatir les colonnes si MultiIndex
spy_ret.columns = [c[-1] if isinstance(c, tuple) else c for c in spy_ret.columns]

# Alignement avec portfolio_df
portfolio_df = portfolio_df.join(spy_ret, how="inner")

print(portfolio_df)

import matplotlib.ticker as mtick

plt.style.use('ggplot')

portfolio_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum())-1

portfolio_cumulative_return[:end_date].plot(figsize=(16,6))

plt.title('Unsupervised Learning Trading Strategy Returns Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()




# Rendements mensuels
monthly_returns = portfolio_df['Strategy Return']

# Nombre de périodes par an (mensuel = 12)
periods_per_year = 12

# Annualised Return (approximation)
annualised_return = monthly_returns.mean() * periods_per_year

# Annualised Volatility
annualised_vol = monthly_returns.std() * np.sqrt(periods_per_year)

# Annualised Sharpe Ratio (taux sans risque = 0)
annualised_sharpe = annualised_return / annualised_vol

# Rendements mensuels SPY
spy_monthly_returns = portfolio_df['SPY']

# Nombre de périodes par an (mensuel = 12)
periods_per_year = 12

# Annualised Return
spy_annualised_return = spy_monthly_returns.mean() * periods_per_year

# Annualised Volatility
spy_annualised_vol = spy_monthly_returns.std() * np.sqrt(periods_per_year)

# Annualised Sharpe Ratio (taux sans risque = 0)
spy_annualised_sharpe = spy_annualised_return / spy_annualised_vol

# Affichage
print("=== SPY Benchmark ===")
print("Annualised Return: {:.2%}".format(spy_annualised_return))
print("Annualised Volatility: {:.2%}".format(spy_annualised_vol))
print("Annualised Sharpe Ratio:", round(spy_annualised_sharpe, 2))

# Comparaison avec ta stratégie
print("\n=== Strategy vs SPY ===")
print("Strategy Annualised Return: {:.2%}".format(annualised_return))
print("Strategy Annualised Volatility: {:.2%}".format(annualised_vol))
print("Strategy Annualised Sharpe: {:.2f}".format(annualised_sharpe))