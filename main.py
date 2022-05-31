from tkinter.tix import X_REGION
import anndata
import scanpy as sc
import numpy as np
from sklearn.decomposition import PCA as pca
import argparse
from kmeans import KMeans
import matplotlib.pyplot as plt

def parse_args():
    parser = argparse.ArgumentParser(description='number of clusters to find')
    parser.add_argument('--n-clusters', type=int,
                        help='number of features to use in a tree',
                        default=2)
    parser.add_argument('--data', type=str, default='./Heart-counts.csv',
                        help='data path')

    a = parser.parse_args()
    return(a.n_clusters, a.data)

def read_data(data_path):
    return anndata.read_csv(data_path)

def preprocess_data(adata: anndata.AnnData, scale :bool=True):
    """Preprocessing dataset: filtering genes/cells, normalization and scaling."""
    sc.pp.filter_cells(adata, min_counts=5000)
    sc.pp.filter_cells(adata, min_genes=500)

    sc.pp.normalize_per_cell(adata, counts_per_cell_after=1e4)
    adata.raw = adata

    sc.pp.log1p(adata)
    if scale:
        sc.pp.scale(adata, max_value=10, zero_center=True)
        adata.X[np.isnan(adata.X)] = 0

    return adata


def PCA(X, num_components: int):
    return pca(num_components).fit_transform(X)

def main():
    n_classifiers, data_path = parse_args()
    heart = read_data(data_path)
    heart = preprocess_data(heart)
    X = PCA(heart.X, 100)

    # Your code
    kmeans = KMeans(7,'random',300)
    clustering = kmeans.fit(X)
    Xplot = PCA(X, 2)
    x = Xplot[:,0]
    y = Xplot[:,1]
    visualize_cluster(x, y, clustering)
    #kmeans.silhouette(clustering, X)


def visualize_cluster(x, y, clustering):
    #Your code
    
    # generate n colors
    color = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

    # plot cluster with different color
    for i in range(x.shape[0]):
        cluster_num = clustering[i]
        plt.plot(x[i], y[i], '.', color = color[cluster_num])

    plt.title("visualization the clusters")
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig('plot.jpg')

    

if __name__ == '__main__':
    main()
