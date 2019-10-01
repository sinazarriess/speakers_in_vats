from gensim.models import word2vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors
import pandas as pd
import numpy as np
from itertools import combinations
from itertools import permutations
from collections import Counter
import gzip
import json
import random
from collections import defaultdict
import sys,os


# train situational embeddings for head nouns
# the context of a word are the nouns referring to other
# objects in the same scene

#path = "/Volumes/SHK-DropBox/dsg-vision/PreprocOut/"
path = "../data/refdf/"


def load_traindf(dfname):

    path1 = path + dfname
    refdf = pd.read_json(path1,compression='gzip', orient='split')

    return refdf


# train standard w2v model on referring expressions
# predict left and right context of a word
def textual(traindf,outfilename,iter=4,dim=100,col='refexp'):

    #if os.path.isfile(outfilename):
    #    print('Outfile (%s) exists. Better check before I overwrite!'\
    #        % (outfilename))
    #    exit()

    allrefs = [ref.split() for ref in traindf[col]]
    w2v_ref = word2vec.Word2Vec(allrefs, size=dim, sg=2, window=5, min_count=5, workers=2, iter=4)
    #of = get_tmpfile(outfilename)
    w2v_ref.wv.save_word2vec_format(outfilename,binary=False)

# train denotational embeddings
# the context of a word are other words used to refer to the same object/region
def denotational(fulldf,outfilename):

    if os.path.isfile(outfilename):
        print('Outfile (%s) exists. Better check before I overwrite!'\
            % (outfilename))
        exit()

    # group the data frame by regions
    rgb = fulldf.groupby(['i_corpus', 'image_id','region_id'])

    # ... simply pair each word with every other word from the same region
    concat_ref = []
    for k in rgb.groups.keys():
        #print k
        reflist = [ref for ref in rgb.get_group(k)['refexp']]
        if len(reflist) > 1:
            for rcomb in combinations(reflist,2):
                #print rcomb
                concat_ref += [(w1,w2) for w1 in rcomb[0].split() for w2 in rcomb[1].split()]
    w2v_den_all = word2vec.Word2Vec(concat_ref, size=100, sg=2, window=1, min_count=5, workers=2, iter=4)
    #of = get_tmpfile(outfilename)
    w2v_den_all.wv.save_word2vec_format(outfilename,binary=False)

def train_on_refcoco():

    refdf = load_traindf("refcoco_refdf.json.gz")
    print("textual refcoco")
    print(len(refdf))
    #textual(refdf,'../data/cbow/refcoco_lr_traindf_100dim_iter20.kv',iter=20,dim=100)
    #print("denotational refcoco")
    #denotational(refdf,'../data/cbow/refcoco_den_traindf_100dim.kv')

    refdf2 = load_traindf("refcocoplus_refdf.json.gz")
    print("textual refcoco+")
    print(len(refdf2))
    #textual(refdf2,'../data/cbow/refcocoplus_lr_traindf_100dim_iter20.kv',iter=20)
    #print("denotational refcoco")
    #denotational(refdf2,'../data/cbow/refcocoplus_den_traindf_100dim.kv')

    refdf3 = load_traindf("grex_refdf.json.gz")
    print(len(refdf3))
    print("textual grex")
    print(refdf3.head())
    #textual(refdf3,'../data/cbow/grex_lr_traindf_100dim_iter20.kv',iter=20)

    refdffull = pd.concat([refdf,refdf2,refdf3])
    print(len(refdffull))
    print("textual refcoco both")
    #textual(refdffull,'../data/cbow/refcocogrex_lr_traindf_200dim_iter10.kv',iter=10,dim=200)
    textual(refdffull,'../data/cbow/refcocogrex_lr_traindf_200dim_iter20.kv',iter=20,dim=200)

def train_on_vg():

    refdf = load_traindf("vgregdf.json.gz")
    print(refdf.head())# WARNING: )
    print(refdf.columns)
    textual(refdf,'../data/cbow/vg_regiondf_100dim_iter20.kv',iter=20,dim=100,col='phrase')


if __name__ == '__main__':


    #train_on_vg()
    train_on_refcoco()

    #print("denotational refcoco both")
    #denotational(refdffull,'../data/cbow/refcocoboth_den_traindf_100dim.kv')
