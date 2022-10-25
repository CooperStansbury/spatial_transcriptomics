import pandas as pd
import numpy as np
import scipy
import os 
import scanpy as sc
import umap
from sklearn.cluster import SpectralClustering
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
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
        print("Done SVD.")
        
        
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
        print("Done OHT.")
        
            
    def UMAP(self, centered=True, **kwargs):
        X = self._getX(centered)
        self.umap_reducer = umap.UMAP(**kwargs)
        self.umap_params = dict(kwargs)
        self.umap_embedding = self.umap_reducer.fit_transform(X)
        print("Done UMAP.")
        
        
    def querylClusters(self):
        """A function to return results of the clustering """
        if self.clusters is None:
            raise ValueError("Clusters not computed, run a clustering method!")
            
        print(f"Querying {self.cluster_type}")
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
        print("Done Querying.")
    
        
    def simpleClustering(self, k):
        """k-means on UMAP embeddings"""
        if self.umap_reducer is None:
            raise ValueError("UMAP not computed, run Data.UMAP()!")
        
        kmeans = KMeans(n_clusters=k,
                        random_state=0).fit(self.umap_embedding)

        self.cluster_type = "simple"
        self.k = k 
        self.clusters = kmeans
        self.labels = kmeans.labels_
        
        print("Done Simple Clustering.")
        
    
    def dotsonClustering(self, k):
        """A function to perform clustering as in the 
        manuscript """
        if self.u is None:
            print("Running SVD...")
            self.svd()
        
        if self.oht is None:
            print("Computing OHT...")
            self.getOHT()
            
        # construct P, note that the data is transposed
        P = self.u[:, 0:self.oht] 
        
        # precompute distance matrix 
        A = scipy.spatial.distance.pdist(P, 'euclidean')
        A = scipy.spatial.distance.squareform(A)
        spect = SpectralClustering(n_clusters=k,
                              assign_labels='discretize',
                              affinity='precomputed', # this is important !
                              random_state=0).fit(A)
        
        self.cluster_type = "dontson"
        self.k = k 
        self.clusters = spect
        self.labels = spect.labels_
        
        


            
        
            
        
        
        
        
        
            
            
        
        
        
        
    