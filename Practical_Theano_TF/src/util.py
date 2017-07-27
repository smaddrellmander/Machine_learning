# util.py
from __future__ import print_function, division
from builtins import range
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression

def get_transformed_data():
    print("Reading in and transforming the data...")
    if not os.path.exists('../large_files/train.csv'):
        print('File not found')
        exit()

    df = pd.read_csv('../large_files/train.csv')
    data = df.as_matrix().astype(np.float32)
    np.random.shuffle(data)

    X = data[:, 1:]
    mu = X.mean(axis=0)
    X = X - mu # Center the data
    pca = PCA() # Principal Component Analysis
    Z = pca.fit_transform(X)
    Y = data[:, 0].astype(np.int32)

    plot_cumulative_variance(pca)

    return Z, Y, pca, mu

def plot_cumulative_variance(pca):
    P = []
    for p in pca.explained_variance_ratio_:
        if len(P) == 0:
            P.append(p)
        else:
            P.append(p + P[-1])
    plt.plot(P)
    plt.ylabel("Cumalative variance")
    plt.show()
    return P

def benchmark_pca():
    print("start:")
    X, Y, _, _ = get_transformed_data()
    pass

if __name__ == '__main__':
    # 
    benchmark_pca()
    #
