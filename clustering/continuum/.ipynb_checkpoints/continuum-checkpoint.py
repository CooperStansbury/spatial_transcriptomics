import os
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
from sklearn import metrics
import itertools
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csgraph
from numpy.polynomial.polynomial import polyfit
import matplotlib.pyplot as plt
import umap
from sklearn import linear_model
from collections import Counter
import sklearn
import scipy.stats as sps
from scipy.spatial.distance import cdist
from importlib import reload



def getGroup(rna, labels, key, ctype):
    """A function to select cells from a group """
    rf = rna[key]
    lf = labels[key]
    lf = lf.drop_duplicates(subset='cellId')
    
    cellIds = lf[lf['cellType'] == ctype]['cellId'].to_list()
    return rf[rf.index.isin(cellIds)]


def getDEGs(p1, p2):
    """ get differential expression between p1, p2
    Note: columns assumed to bet the same
    """
    deg = []
    for g in p1.columns:
        score, pval = scipy.stats.ranksums(p1[g], p2[g],
                                   alternative='two-sided')

        lfc = np.log2(p1[g].mean()+0.001) - np.log2(p2[g].mean()+0.001)

        row = {
            'gene' : g,
            'score' : score,
            'pval' : pval,
            'log2foldchange' : lfc,
            'meanP1' : p1[g].mean(),
            'propP1' : p1[g].astype(bool).sum() / len(p1),
            'meanP2' : p2[g].mean(),
            'propP2' : p2[g].astype(bool).sum() / len(p2),
        }

        deg.append(row)
    deg = pd.DataFrame(deg)
    return deg 


def getSignificant(deg, alpha, n, p=None, method='log2foldchange'):
    """get significant genes from deg 
    
    alpha: the significance level before correction
    n: number of genes for each group 
    p: the percentage of cells that must express the gene (from each group)
    """
    ntests = deg['gene'].nunique()
    alphaHat = 1 - ((1-alpha) ** (1/ntests))
    print(f"{alpha=} {ntests=} {alphaHat}")


    sig = deg[deg['pval'] < alphaHat].reset_index()
    
    if not p is None:
        sig = sig[(sig['propP1'] > p) | (sig['propP2'] > p)]

    """ gene sorting method """
    if method == 'log2foldchange': # sort by high-low logfold change
        n = int(n/2) # taking from top and bottom 
        sig = sig.sort_values(by='log2foldchange', ascending=False)
        sig = pd.concat([sig.head(n), sig.tail(n)])
    elif method == "pval": # sort by pval as in macspectrum
        sig = sig.sort_values(by='pval', ascending=True)
        sig = sig.head(n) 
    else: 
        raise NotImplementedError('No other methods')

    # extract the DEGs
    corrGenes = sig['gene'].to_list()
    
    return corrGenes, sig


def getSignature(p, genes, norm=False, log=True):
    """A function to get the aggregate signature """
    signature = p[genes].mean(axis=0).to_numpy()
    signature = signature.reshape((1, len(signature))) 
    signature = signature.astype(float)
    
    if log:
        signature = np.log1p(signature)
    
    if norm:
        signature = sklearn.preprocessing.minmax_scale(signature, 
                                                       feature_range=(0, 1), 
                                                       axis=1)
        
    return signature


def distanceFromAnchor(df):
    """a function to compute the distance from the upper left-most point"""
    upperLeft = df[df['p2_on_fit'] ==  df['p2_on_fit'].max()]
    upperLeft = upperLeft.sort_values(by='p2', ascending=False)
    upperLeft = upperLeft.head(1)
    
    xMax = upperLeft['p1'].values[0]
    yMax = upperLeft['p2_on_fit'].values[0]
    
    anchor = np.array([xMax, yMax])
    anchor = anchor.reshape((1, len(anchor)))
    
    dists = scipy.spatial.distance.cdist(df[['p1', 'p2_on_fit']], 
                                         anchor, 
                                         metric='euclidean')
    
    dists = sklearn.preprocessing.minmax_scale(dists, 
                                               feature_range=(0, 1), 
                                               axis=0)
    return dists

def getProjection(pdf):
    """A function to learn p1/p2 polarization """
    # define variables
    X = pdf['p1'].values[:,np.newaxis]
    y = pdf['p2'].values
    
    # fit the optimal line, get coefficients
    model = linear_model.LinearRegression()
    
    model.fit(X, y)
    m = model.coef_[0]
    b = model.intercept_

    print(f"x = p1, y = p2")
    print(f"{-m:.4f}x + y + {-b:.4f} = 0")
    
    # add predicted values
    fit = model.predict(X)
    pdf['fit'] = fit
    
    # perform macSpectrum porjection
    pdf['xp'] = (pdf['p1'] - (-m*pdf['p2']) - (-m * -b)) / ((-m**2 * -b**2))
    pdf['yp'] = ((-m**2 * pdf['p2']) - (-m * pdf['p1']) - (-b)) / ((-m**2 * -b**2))
    
    # get anchor on p1
    x0 = pdf['p1'].min()
    y0 = pdf['fit'].min()
    # compute macSpectrum distance
    l = np.sqrt((pdf['xp'] - x0)**2 + (pdf['yp'] - y0)**2 )
    pdf['l'] = l
    
    # simple projection onto the OLS solution
    p2_on_fit = (np.dot(y, fit)/np.sqrt(sum(fit**2))  **2)*fit
    pdf['p2_on_fit'] = p2_on_fit
    
    pdf['dists'] = distanceFromAnchor(pdf)
    
    return pdf


def filterPoints(df,lb, ub):
    """A function to filter points based on the distribution """
    mask1 = (df['p1'] > df['p1'].quantile(lb)) & ((df['p1'] < df['p1'].quantile(ub)))
    df = df[mask1]
    
    mask2 = (df['p2'] > df['p2'].quantile(lb)) & ((df['p2'] < df['p2'].quantile(ub)))
    df = df[mask2]
    return df.reset_index(drop=True)