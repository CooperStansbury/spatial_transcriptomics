import pandas as pd
import numpy as np
import scipy
import os 
import scanpy as sc
import umap
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics
from importlib import reload
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns



class Data:
    def __init__(self, df, dataname):
        self.name = dataname
        self.X = df.T # genes x cell
        self.Xc = self._center()
        
        # empty slots
        self.u = None
        self.s = None
        self.vh = None
        self.umap_reducer = None
        self.umap_params = None
        self.umap_embedding = None
        self.oht = None
        
        # clustering attribs
        self.cluster_type = None
        self.cluster_data = None
        self.k = None
        self.clusters = None
        self.labels = None
        self.results = None
        
        
    def _center(self):
        Xc = self.X.apply(lambda x: x-x.mean())
        return Xc
        
        
    def _getX(self, centered=True):
        if centered:
            X = self.Xc
        else:
            X = self.X
        return X
    
    
    def svd(self, centered=True):
        X = self._getX(centered)
        u, s, vh = np.linalg.svd(X)    
        self.u = u
        self.s = s
        self.vh = vh  
        
        
    def getOHT(self):
        if self.u is None:
            raise ValueError("SVD not computed, run Data.svd()!")
            
        m = self.u.shape[0]
        n = self.vh.shape[0] 
        beta = m / n
        omega = (0.56*beta**3) - (0.95 * beta**2) + (1.82 * beta) + 1.43
        y_med = np.median(self.s)
        tau = omega * y_med
        s_ind = np.argwhere(self.s >= tau)
        self.oht = np.max(s_ind) 
        
            
    def UMAP(self, centered=True, **kwargs):
        X = self._getX(centered)
        self.umap_reducer = umap.UMAP(**kwargs)
        self.umap_params = dict(kwargs)
        self.umap_embedding = self.umap_reducer.fit_transform(X)
        
        
    def querylClusters(self):
        """A function to return results of the clustering """
        if self.clusters is None:
            raise ValueError("Clusters not computed, run a clustering method!")
            
        res = {}
        for label in set(self.labels):
            data = self.X[self.labels == label].T
            
            grped = data.agg(['sum', 
                            'mean',
                            'nunique',
                            'std', 
                            'median', 
                            'min', 
                            'max'], axis=1)

            grped['cluster'] = label
            grped['nCells'] = data.shape[1]
            res[label] = grped.reset_index(drop=False) 
        self.results = res
        

    def scoreClusters(self):
        if self.clusters is None:
            raise ValueError("Clusters not computed, run a clustering method!")
            
        ss = metrics.silhouette_score(self.cluster_data, 
                                      self.labels, 
                                      metric='euclidean')
        
        vrc = metrics.calinski_harabasz_score(self.cluster_data, 
                                              self.labels)
        
        db = metrics.davies_bouldin_score(self.cluster_data, 
                                              self.labels)
        
        return {
            'silhouette_score' : ss,
            'calinski_harabasz_score' : vrc,
            'davies_bouldin_score' : db
        }
        
    def simpleClustering(self, k, n=2, use_umap=True):
        """k-means on UMAP embeddings
        k is the number of clusters
        n is the number of umap dimensions used
        """
        if self.umap_reducer is None:
            raise ValueError("UMAP not computed, run Data.UMAP()!")
            
        if use_umap:
            self.cluster_data = self.umap_embedding[:, 0:n]
        else:
            self.cluster_data = self.u[:, 0:n]
        
        kmeans = KMeans(n_clusters=k,
                        random_state=0).fit(self.cluster_data)

        self.cluster_type = "simple"
        self.k = k 
        self.clusters = kmeans
        self.labels = kmeans.labels_
        
        
    
    def dotsonClustering(self, k):
        """A function to perform clustering as in the 
        manuscript """
        if self.u is None:
            self.svd()
        
        if self.oht is None:
            self.getOHT()
            
        # construct P, note that the data is transposed
        P = self.u[:, 0:self.oht] 
        
        # precompute distance matrix 
        A = scipy.spatial.distance.pdist(P, 'euclidean')
        self.cluster_data = scipy.spatial.distance.squareform(A)
        spect = SpectralClustering(n_clusters=k,
                              assign_labels='discretize',
                              affinity='precomputed', # this is important !
                              random_state=0).fit(self.cluster_data)
        
        self.cluster_type = "dontson"
        self.k = k 
        self.clusters = spect
        self.labels = spect.labels_
        
        


            
        
            
        
        
        
        
        
            
            
        
        
        
        
    