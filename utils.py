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

df = pd.read_csv(train_file, index_col='id')
y = df.sales
x = df.drop('sales', axis=1)


def print_histogram(data, feature=None):
    sns.histplot(data, y=feature)
    plt.title(f"histogram of {feature}")
    plt.show()


def vanilla_encoder(x, feature):
    enc = OrdinalEncoder()
    enc.fit(x[feature].values.reshape(-1,1))
    print(enc.categories_)
    x[feature] = enc.transform(x[feature].values.reshape(-1,1))


vanilla_encoder(x, 'family')
print(x.family.unique())
# plt.plot(x=x, y=y)
