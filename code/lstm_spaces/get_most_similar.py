#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 17 2019

@author: Sina Zarrieß
"""

import numpy as np
import pickle
import json
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from gensim.models import word2vec


categories = [20]

words = ['horse','man','blue','food','orange','funny',
'broccoli']

def print_most_similar(simmat,word2index,index2word):

    for w in words:
        windex = word2index[w]
        neighbours = np.argsort(simmat[windex])
        print(windex, w,":\n",
        "nearest:",[index2word[nw] for nw in neighbours[:5]],"\n",
        "similarities:",[simmat[windex][nw] for nw in neighbours[:5]])

for cat in categories:
    print("Model,",cat)

    for corp in ['rc','rcplus']:
        print("Data:",corp)
        emfile = "../data/embeddings/embeddings_%s_region_model%d.npz"%(corp,cat)
        emb = np.load(emfile)['arr_0']
        #print(emb.shape)


        wfile = "../data/embeddings/wordids_%s_region_model%d.json"%(corp,cat)
        with open(wfile,"r") as f:
            w2id = json.load(f)

        id2w = {w2id[w]:w for w in w2id}
        #print(len(w2id))

        sim = squareform(pdist(emb,metric="cosine"))
        #print(sim.shape)

        print_most_similar(sim,w2id,id2w)

ref = word2vec.Word2Vec.load('../data/cbow/refcocoplus_den_traindf_100dim.mod')
#w2v_den = word2vec.Word2Vec.load('w2v_den_headonly_300dim.mod')
refplus = word2vec.Word2Vec.load('../data/cbow/refcoco_den_traindf_100dim.mod')
