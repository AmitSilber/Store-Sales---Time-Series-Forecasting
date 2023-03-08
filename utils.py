import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as dates
import matplotlib as mpl

import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_log_error, mean_squared_error

# import gensim.downloader as api
# from gensim.models import FastText

train_file = 'train.csv'


def print_histogram(data):
    df_melted = data.melt(var_name='column')
    g = sns.FacetGrid(df_melted, row='column')
    g.map(plt.hist, 'value')
    plt.show()


def temporal_date(data):
    data.date = pd.to_datetime(data.date)
    data.date = data.date.dt.to_pydatetime()
    data.date = dates.date2num(data.date)
    return data


def cluster_family_by_sales_corr(df):
    plt.figure(figsize=(16, 7))
    sales = pd.DataFrame({g: s.values for g, s in df.groupby('family').sales})
    corr = sales.corr()
    sns.clustermap(corr, cmap='Blues')
    plt.savefig('family_corr.png')


def print_histogram(data, feature=None):
    sns.histplot(data, y=feature)
    plt.title(f"histogram of {feature}")
    plt.show()


def vanilla_encoder(x, feature):
    enc = OrdinalEncoder()
    enc.fit(x[feature].values.reshape(-1, 1))
    x[feature] = enc.transform(x[feature].values.reshape(-1, 1))


def vanilla_model(X_test, X_train, y_test, y_train):
    model = RandomForestRegressor(n_estimators=10, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmsle = mean_squared_log_error(y_test, y_pred)
    print(rmsle)


def main():
    df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    y = df.sales
    x = df.drop('sales', axis=1)
    x = temporal_date(x)
    vanilla_encoder(x, 'family')
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    # # z = np.polyfit(x, y)
    # mpl.rcParams['agg.path.chunksize'] = 10000
    # sns.relplot(data=X_train, x='store_nbr', y=y_train, col='family', hue='onpromotion')
    # plt.show()
    vanilla_model(X_test, X_train, y_test, y_train)


if __name__ == '__main__':
    main()
