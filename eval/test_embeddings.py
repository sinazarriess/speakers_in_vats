import sys
import numpy as np
from os import listdir
from os.path import isfile, join
import matplotlib.pyplot as plt
from utils import read_json_vocab, compute_PCA, print_matrix, print_vocab
from evals import RSA, compute_cosines, compute_men_spearman


def get_common_words(vocab_files):
    vocabs = []

    for v in vocab_files:
        d = read_json_vocab(v)
        vocabs.append(set(d.keys()))

    intersection = set.intersection(*vocabs)
    print(len(intersection),"common words in vocabulary")
    return sorted(list(intersection))

def get_common_word_indices(vocab_file,common_words):
    indices = []
    vocab = read_json_vocab(vocab_file)
    for w in common_words:
        indices.append(vocab[w])    #indices has the order of common_words -- important for comparing speakers
    return indices

def slice_matrix_in_order(m,indices):
    a = []
    for i in indices:
        a.append(m[i])
    return np.array(a)

def compute_cosine_matrices(speaker_files):
    speakers = []

    for s in speaker_files:
        print(s)
        vocab_file = s.replace("embeddings_","wordids_").replace(".npz",".json")
        indices = get_common_word_indices(vocab_file,common_words)
        m = np.load(s)['arr_0']
        m = slice_matrix_in_order(m,indices)
        print_matrix(m,common_words,s.replace("embeddings_","common_vocab_").replace(".npz",".dm"))
        m_cos = compute_cosines(m)
        speakers.append(m_cos)
        print_matrix(m_cos,common_words,s.replace("embeddings_","cosines_").replace(".npz",".dm"))
    return speakers



spacedir = '../data/embeddings/'
speaker_files = [join(spacedir, f) for f in listdir(spacedir) if ".npz" in join(spacedir, f)]
print("Found",len(speaker_files),"speaker files...")

vocab_files = [join(spacedir, f) for f in listdir(spacedir) if ".json" in join(spacedir, f)]
print("Found",len(vocab_files),"vocab files...")

common_words = get_common_words(vocab_files)
print_vocab(common_words,join(spacedir,"common_vocab.txt"))

speakers = compute_cosine_matrices(speaker_files)
rsa_matrix = np.zeros((len(speakers),len(speakers)))

for i in range(len(speakers)):
    for j in range(len(speakers)):
        print("Computing RSA",i,j)
        rsa_matrix[i][j] = RSA(speakers[i],speakers[j])[0]
        print(rsa_matrix[i][j])

print(rsa_matrix)
red_rsa_matrix = compute_PCA(rsa_matrix,2)
print(red_rsa_matrix)
make_figure(red_rsa_matrix)

