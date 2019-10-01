#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sep 17 2019

@author: Sina Zarrieß
"""

import numpy as np
from tensorflow.python import pywrap_tensorflow
import pickle
import json
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file

# ***********
# I used this script to load the embedding layer of some LSTMS which were trained with tensorflow (ACL 2019 paper)
# ... and save this layer as an np file
# the LSTMs were trained on refcoco (rc)
# files are named according to which category was excluded from the training set
# ***********


#path = "/Users/sina/research/bielefeld/sinaza/041_refrnn/trainable_decoding/zero_lstm/"
path = "/Users/sina/research/bielefeld/sinaza_citec_backup/041_refrnn/trainable_decoding/region_diff_lstm/"
# these are IDs of MSCOCO categories
# they mean something like cup, bottle, cat, dog, ...
#categories = [17,19,44,47,6,7]
categories = [5,10,15,20]

# ID of the model's checkpoint which will be loaded
model_ids = [5,10,15,20] #5

def save_embeddings(checkfile,indexfile,outfile):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint)
    emb = reader.get_tensor("word_embedding/w")
    #outfile = "../data/embeddings/embeddings_rc_%dsplit.npz"%cat
    #np.savez(embfile,emb)
    #print(emb.shape)

    with open(indexfile, 'rb') as f:
        w2id = pickle.load(f)
    print(len(w2id))
    print(list(w2id.items())[:10])
    #outfile = "../data/embeddings/wordids_rc_%dsplit.json"%cat
    #with open(wordfile,"w") as f:
    #    json.dump(w2id,f)

    f = open(outfile,"w")
    f.write("%d %d\n"%emb.shape)
    for word in w2id:
        vec = ' '.join([word]+[str(i) for i in emb[w2id[word]]])
        f.write(vec+"\n")
    f.close()

    return True

def print_tensors(checkfile):
    print_tensors_in_checkpoint_file(checkfile, all_tensors=True, tensor_name='')

    reader = pywrap_tensorflow.NewCheckpointReader(checkfile)
    emb = reader.get_tensor("word_embedding/w")
    print("Embedding")
    print(emb)
    print(emb.shape)


for mid in model_ids:
    print("Model,",mid)
    #checkpoint = path + "rc_%dsplit/model/model-%d"%(cat,model_id)
    #o1 = "../data/embeddings/embeddings_rc_%dsplit.npz"%cat
    #checkpoint = path + "rcplus_%dsplit/model/model-%d"%(cat,model_id)
    #o1 = "../data/embeddings/embeddings_rcplus_%dsplit.npz"%cat

    #wordindex = path + "rcplus_%dsplit/data/word_to_idx.pkl"%cat
    #o2 = "../data/embeddings/wordids_rcplus_%dsplit.json"%cat

    #wordindex = path + "rc_%dsplit/data/word_to_idx.pkl"%cat
    #o2 = "../data/embeddings/wordids_rc_%dsplit.json"%cat


    checkpoint = path + "models/rc_region/model-%d"%(mid)
    wordindex = path + "refcocodata/train/word_to_idx.pkl"
    o = "../data/reg_spaces/refcoco_region_model%d.kv"%mid


    save_embeddings(checkpoint,wordindex,o)


    checkpoint = path + "models/rcplus_region/model-%d"%(mid)
    wordindex = path + "refcocoplusdata/train/word_to_idx.pkl"
    o = "../data/reg_spaces/refcocoplus_region_model%d.kv"%mid

    save_embeddings(checkpoint,wordindex,o)



    #print_tensors(checkpoint)

#checkpoint = path + "rc_listenernolab/model/model-%d"%(model_id)
#o1 = "../data/embeddings/embeddings_rc_0split.npz"
#wordindex = path + "rc_listenernolab/data/word_to_idx.pkl"
#o2 = "../data/embeddings/wordids_rc_0split.json"
#save_embeddings(checkpoint,wordindex,o1,o2)
