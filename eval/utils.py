import random
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine
from sklearn.decomposition import PCA
from sklearn import preprocessing
from scipy import linalg as LA
import numpy as np
import json


def normalise_l1(m):
    return preprocessing.normalize(m, norm='l1')

def normalise_l2(m):
    return preprocessing.normalize(m, norm='l2')

def compute_cosines(m):
    return 1-pairwise_distances(m, metric="cosine")

def ppmi(m):
    ppmi_matrix = np.zeros(m.shape)
    N = np.sum(m)
    row_sums = np.sum(m, axis=1)
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ppmi_matrix[i][j] = max(0, m[i][j] * N / (row_sums[i] * row_sums[j]))
    return ppmi_matrix

def compute_PCA(m,dim):
    m -= np.mean(m, axis = 0)
    pca = PCA(n_components=dim)
    pca.fit(m)
    return pca.transform(m)

def read_external_vectors(vector_file):
    vocab = []
    vectors = []
    with open(vector_file) as f:
        dmlines=f.read().splitlines()
    for l in dmlines:
        items=l.split()
        target = items[0]
        vocab.append(target)
        vec=[float(i) for i in items[1:]]       #list of lists  
        vectors.append(vec)
    m = np.array(vectors)
    return m, vocab

def read_json_vocab(vocab_file):
    with open(vocab_file) as vocab:
        d = json.load(vocab)
    return(d)
    

def print_matrix(dm_mat,vocab,outfile):
    '''Print new dm file'''
    f = open(outfile,'w')
    for c in range(dm_mat.shape[0]):
        vec = ' '.join([str(i) for i in dm_mat[c]])
        f.write(vocab[c]+" "+vec+"\n")
    f.close()

def print_vocab(vocab,outfile):
    '''Print new vocab file'''
    f = open(outfile,'w')
    for c in range(len(vocab)):
        line = str(c)+' '+vocab[c]+'\n'
        f.write(line)
    f.close()

def print_dict(d,outfile):
    f = open(outfile,'w')
    for k,v in d.items():
        line = str(k)+' '+str(v)+'\n'
        f.write(line)
    f.close()

def print_list(l,outfile):
    f = open(outfile,'w')
    for v in l:
        f.write(str(v)+'\n')
    f.close()
