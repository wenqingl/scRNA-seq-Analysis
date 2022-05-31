from turtle import update
import numpy as np
import random

class KMeans():

    def __init__(self, n_clusters: int, init: str='random', max_iter = 300):
        """

        :param n_clusters: number of clusters
        :param init: centroid initialization method. Should be either 'random' or 'kmeans++'
        :param max_iter: maximum number of iterations
        """
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.centroids = None # Initialized in initialize_centroids()

    def fit(self, X: np.ndarray):
        self.initialize_centroids(X)
        iteration = 0
        clustering = np.zeros(X.shape[0])
        while iteration < self.max_iter:
            # your code
            
            # assign cluster
            dist = self.euclidean_distance(X, self.centroids)
            clustering = np.argmin(dist, axis=1)
            self.update_centroids(clustering, X)
            iteration = iteration + 1
        #np.set_printoptions(threshold=np.inf)
        #print(clustering)   
        return clustering

    def update_centroids(self, clustering: np.ndarray, X: np.ndarray):
        #your code
        Xrow, Xcolumn = X.shape
        sum = np.zeros((self.n_clusters, Xcolumn))
        num = np.zeros(self.n_clusters)
        for i in range(Xrow):
            cluster_num = clustering[i]
            sum[cluster_num] = sum[cluster_num] + X[i]
            num[cluster_num] = num[cluster_num] + 1
        
        for i in range(self.n_clusters):
            temp = sum[i] / num[i]
            self.centroids[i] = temp

        

    def initialize_centroids(self, X: np.ndarray):
        """
        Initialize centroids either randomly or using kmeans++ method of initialization.
        :param X:
        :return:
        """
        if self.init == 'random':
            random_indices = np.random.choice(X.shape[0], size=self.n_clusters, replace=False)
            self.centroids = X[random_indices, :]
            
        elif self.init == 'kmeans++':
            # initial
            Xrow = X.shape[0]
            self.centroids = np.zeros((1, X.shape[1]))

            # randomly choose a centroids
            random_indices = np.random.choice(Xrow, size=1, replace=False)
            self.centroids = np.array(X[random_indices])
            centroid_num = 1

            while(centroid_num  < self.n_clusters):
                # compute the dist for every point 
                dist = self.euclidean_distance(X,self.centroids)

                # initial D
                each_D = dist.min(axis=1)
                total_D2 = sum(i*i for i in each_D)

                # find the largest prob and store into centroids
                largest_pro = -1
                pro = np.zeros(Xrow)
                for i in range (Xrow):
                    pro[i] = each_D[i]**2 / total_D2
                    '''if(each_D[i]**2 / total_D2 > largest_pro):
                        largest_pro = each_D[i]**2 / total_D2
                        largest_pro_index = i'''
                
                #b = np.array([X[largest_pro_index]])
                sampleNumbers = np.random.choice(list(range(0,Xrow)), 1, p=pro)
                self.centroids = np.r_[self.centroids,X[sampleNumbers]]
                centroid_num = centroid_num + 1

        else:
            raise ValueError('Centroid initialization method should either be "random" or "k-means++"')

    def euclidean_distance(self, X1:np.ndarray, X2:np.ndarray):
        """
        Computes the euclidean distance between all pairs (x,y) where x is a row in X1 and y is a row in X2.
        Tip: Using vectorized operations can hugely improve the efficiency here.
        :param X1:
        :param X2:
        :return: Returns a matrix `dist` where `dist_ij` is the distance between row i in X1 and row j in X2.
        """
        # your code
        # get the size of x1 and x2
        X1row = X1.shape[0]
        X2row = X2.shape[0]

        # initial matrix with 0
        dist = np.zeros((X1row, X2row))

        # calculate the dist and strore in matrix
        for i in range(X1row):
            for j in range(X2row):
                dist[i][j] = np.linalg.norm(X1[i] - X2[j])

        return dist

    def silhouette(self, clustering: np.ndarray, X: np.ndarray):
        # your code
        dist = self.euclidean_distance(X, self.centroids)
        best = np.min(dist, axis = 1)
        best_index = np.argmin(dist, axis = 1)

        Xrow = X.shape[0]
        for i in range(Xrow):
            colum = best_index[i]
            dist[i][colum] = float("inf")
        
        second_best = np.min(dist, axis = 1)

        s = np.zeros(Xrow)
        for i in range(Xrow):
            divisor = max(best[i], second_best[i])
            s[i] = (second_best[i] - best[i]) / divisor

        print(s.sum()/Xrow)
        return  s.sum()/Xrow


