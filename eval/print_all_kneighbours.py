import sys
from utils import read_common_vocab, read_cosines, compute_nearest_neighbours, print_nearest_neighbours

speaker = sys.argv[1]

common_words = read_common_vocab()
print("Loading cosines...")
cosines = read_cosines(speaker)

nns = compute_nearest_neighbours(cosines,common_words)
print_nearest_neighbours(nns,speaker.replace('cosines','nns'))

