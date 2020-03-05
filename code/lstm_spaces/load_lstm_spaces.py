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


categories = [17,19,44,47,6,7,0]


for cat in categories:
    print("Category,",cat)
    emfile = "../data/embeddings/embeddings_rc_%dsplit.npz"%cat
    emb = np.load(emfile)['arr_0']
    print(emb.shape)


    wfile = "../data/embeddings/wordids_rc_%dsplit.json"%cat
    with open(wfile,"r") as f:
        w2id = json.load(f)
    print(len(w2id))

    v1 = emb[[w2id['dog'],w2id['cow'],w2id['horse']]]
    print(cosine_similarity(v1))
