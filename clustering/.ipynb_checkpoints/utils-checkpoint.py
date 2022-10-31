import os
import pandas as pd
import numpy as np
import matplotlib 
import matplotlib.pyplot as plt
import scanpy as sc


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


def scoreCluster(geneClusterRanks, cellTypeSig, use_mean_rank=False):
    """A function to score a cluster based on within-cluster 
    gene rankings and a list of input marker genes 

    Args:
        geneClusterRanks (pd.Dataframe): gene names and ranks, must
            have columns: ['gene', 'rank']
        cellTypeSig (list or array): marker genes for the cell type
    """
    matches = geneClusterRanks[geneClusterRanks['gene'].isin(cellTypeSig)]
    if use_mean_rank:
        s = matches['rank'].mean()
    else:
        s = matches['rank'].median()
    return s


