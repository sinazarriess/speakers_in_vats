## Evaluation of speaker spaces

### Representational Similarity Analysis

Running *python3 test_embeddings.py* does the following:

* compute a common vocabulary between all speakers (without which no RSA is possible);
* save embeddings for each speaker as a readable .dm file containing *only* the common words, sorted alphabetically for all speakers;
* compute cosine matrices for each speaker;
* use computed cosine matrices to perform RSA over pairs of speakers.


### Correlation with MEN

*MEN.py* can be run over individual matrices, i.e.: 

    `python3 MEN.py ../data/embeddings/common_vocab_rc_0split.dm`

It returns a Spearman rho figure, and the number of MEN pairs considered in the calculation. (The embedding spaces do not cover the entire MEN vocabulary.)
