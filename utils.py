import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.stats import pearsonr
from scipy.spatial.distance import cdist
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA

import gensim.downloader as api
from gensim.models import FastText

VANILLA = 1
PCA_CORR = 2
UMAP = 3
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


def vanilla_encoder(df, feature):
    enc = OrdinalEncoder()
    enc.fit(df[feature].values.reshape(-1, 1))
    df[feature] = enc.transform(df[feature].values.reshape(-1, 1))


def corr_encoder(df, feature, plot_flag=False):
    sales_per_category = pd.DataFrame({g: s.values for g, s in df.groupby(feature).sales})
    categories = sales_per_category.columns
    corr = sales_per_category.corr()
    pca = PCA(n_components='mle', svd_solver='full')
    encoded_categories = pca.fit_transform(corr)
    if plot_flag:
        plot_encoding(encoded_categories, categories, 3)
        encoded_dist_map(encoded_categories, categories)
    encoded_categories = [",".join(vec.astype(str)) for vec in encoded_categories]
    temp = df[feature].replace(to_replace=categories, value=encoded_categories)
    df['temp'] = temp
    df[['d1', 'd2', 'd3']] = df.temp.str.split(',', expand=True)
    df.drop('temp', axis=1, inplace=True)


def encoding(df, feature, mode):
    if mode == VANILLA:
        vanilla_encoder(df, feature)
    elif mode == PCA_CORR:
        corr_encoder(df, feature)


def encoded_dist_map(encoding, categories):
    dist = 1 / (1 + cdist(encoding, encoding))
    dist = pd.DataFrame(dist, columns=categories, index=categories)
    sns.clustermap(dist, cmap='Blues')
    plt.savefig('family_dist_after_pca.png')


def plot_encoding(encoding, categories, dim):
    fig = plt.figure(figsize=(12, 12))
    if dim == 3:
        ax = fig.add_subplot(projection='3d')
    else:
        ax = fig.add_subplot()
    for i in range(len(encoding)):
        ax.scatter(*list(encoding[i, :]), color='b')
        ax.text(*list(encoding[i, :]), '%s' % (categories[i]),
                size=10, zorder=1,
                color='k')

    plt.savefig(f'family_encoding_{dim}d.png')


def main():
    df = pd.read_csv(train_file, index_col=0, parse_dates=True)
    y = df.sales
    x = df.drop('sales', axis=1)
    x = temporal_date(x)
    encoding(df, 'family', PCA_CORR)
    pd.set_option('display.max_columns', None)
    print(df.head())


if __name__ == '__main__':
    main()
