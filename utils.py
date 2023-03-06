import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

train_file = 'train.csv'

df = pd.read_csv(train_file, index_col='id')
y = df.sales
x = df.drop('sales', axis=1)


def print_histogram(data, feature=None):
    sns.histplot(data,y = feature)
    plt.title(f"histogram of {feature}")
    plt.show()


print(x.family.nunique())
print(x.groupby('store_nbr').family.size())

#plt.plot(x=x, y=y)