import os
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import scanpy as sc
import scipy
from scipy import stats
from Bio import SeqIO
from Bio.KEGG import REST
from Bio.KEGG.KGML import KGML_parser
import io


def read_panglaodb(path):
    """A function to read all panglaodb files """
    df = pd.read_csv(path, 
                     sep='\t',
                     compression='gzip')        
    df['gene'] = df['official gene symbol'].str.upper()
    return df


def getOverlappingGenes(genelist, panglaodf):
    """a function to return a lisr of genes in both """
    overlap = []
    for gene in panglaodf['official gene symbol'].to_list():
        if gene in genelist:
            overlap.append(gene)
    return overlap


def ncolor(n, cmap='viridis'):
    cmap = matplotlib.cm.get_cmap(cmap)
    arr = np.linspace(0, 1, n)
    return [matplotlib.colors.rgb2hex(cmap(x)) for x in arr] 




def getScores(label, clusterGenes, pandf, controlList):
    newRows = []
    nGenes = len(clusterGenes)
    
    for ctype in controlList:
        c = pandf[pandf['cell type'] == ctype]
        ctypeGenes = c['gene'].to_list()
        
        matches = clusterGenes[clusterGenes['gene'].isin(ctypeGenes)]
        s = matches['nRank'].sum() / len(matches)
        
        newCol = f"cluster {int(label)+1}"
            
        row = {
            'type' : ctype,
            newCol : s
        }
        newRows.append(row)

    scores = pd.DataFrame(newRows)
    scores = scores.set_index('type')
    scores = scores.sort_values(by=newCol, ascending=False)
    return scores


def scoreCluster(geneClusterRanks, cellTypeSig, use='mean'):
    """A function to score a cluster based on within-cluster 
    gene rankings and a list of input marker genes 

    Args:
        geneClusterRanks (pd.Dataframe): gene names and ranks, must
            have columns: ['gene', 'rank']
        cellTypeSig (list or array): marker genes for the cell type
    """
    matches = geneClusterRanks[geneClusterRanks['gene'].isin(cellTypeSig)].reset_index()
    
    if use == 'mean':
        s = matches['rank'].mean()
    elif use == 'median':
        s = matches['rank'].median()
    elif use == 'hmean':
        s = stats.hmean(matches['rank'].to_list())
    elif use == 'sum':
        s = matches['rank'].sum()
    else:
        raise ValueError(f'use type {use} undefined')
        
    return s

def parseKEGG(pathId):
    genes = []
    results = REST.kegg_get(pathId).read()
    current_section = None
    for line in results.rstrip().split("\n"):
        section = line[:12].strip()  # section names are within 12 columns
        if not section == "":
            current_section = section

        if current_section == "GENE":
            linesplit = line[12:].split("; ")
            gene_identifiers = linesplit[0]
            gene_id, gene_symbol = gene_identifiers.split()
    
            if not gene_symbol in genes:
                genes.append(gene_symbol)
    return genes

def getPathname(pathId):
    """A function to return the legg pathname"""
    result = REST.kegg_list(pathId).read()
    return result.split("\t")[1].split("-")[0].strip()



def makeColorbar(cmap, width, hieght, title, orientation, tickLabels):
    a = np.array([[0,1]])
    plt.figure(figsize=(width, hieght))
    img = plt.imshow(a, cmap=cmap)
    plt.gca().set_visible(False)
    cax = plt.axes([0.1, 0.2, 0.8, 0.6])
    ticks = np.linspace(0,1 , len(tickLabels))
    cbar = plt.colorbar(orientation=orientation, 
                        cax=cax, 
                        label=title,
                        ticks=ticks)

    if orientation == 'vertical':
        cbar.ax.set_yticklabels(tickLabels)
    else:
        cbar.ax.set_xticklabels(tickLabels)

        
def _normalize_data(X, counts, after=None, copy=False):
    X = X.copy() if copy else X
    if issubclass(X.dtype.type, (int, np.integer)):
        X = X.astype(np.float32)  # TODO: Check if float64 should be used
    else:
        counts_greater_than_zero = counts[counts > 0]

    after = np.median(counts_greater_than_zero, axis=0) if after is None else after
    counts += counts == 0
    counts = counts / after
    if scipy.sparse.issparse(X):
        sparsefuncs.inplace_row_scale(X, 1 / counts)
    elif isinstance(counts, np.ndarray):
        np.divide(X, counts[:, None], out=X)
    else:
        X = np.divide(X, counts[:, None])  # dask does not support kwarg "out"
    return X


def normalize(df, target_sum=1):
    """A function to normalize spots """
    index = df.index
    columns = df.columns
    X = df.to_numpy().copy()
    counts_per_cell = X.sum(1)
    counts_per_cell = np.ravel(counts_per_cell)
    cell_subset = counts_per_cell > 0
    Xnorm = _normalize_data(X, counts_per_cell, target_sum)
    
    ndf = pd.DataFrame(Xnorm, columns=columns, index=index)
    return ndf

