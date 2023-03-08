import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
from sklearn.preprocessing import OrdinalEncoder

import gensim.downloader as api
from gensim.models import FastText

train_file = 'train.csv'


def print_histogram(data):
    df_melted = data.melt(var_name='column')
    g = sns.FacetGrid(df_melted, row='column')
    g.map(plt.hist, 'value')
    plt.show()


def temporal_date(data):
    data.date = pd.to_datetime(data.date)
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


def main():
    df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    y = df.sales
    x = df.drop('sales', axis=1)
    x = temporal_date(x)
    vanilla_encoder(x, 'family')

if __name__ == '__main__':
    main()
