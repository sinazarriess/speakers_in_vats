

## Do you copy me?

# Code

* `load_lstm_spaces.py`: simply shows how to load the embeddings files in the data dir

# Data

* embeddings: numpy files with embedding matrices (extracted from tensorflow checkpoints)
* word ids: json files that map words to ids, these ids correspond to the row index in the corresponding embedding matrix
* `embeddings_rc_[ID]split.npz`: this the word embedding layer learnt by an LSTM trained on RefCoco referring expressions, the ID of the split refers to which category has been EXCLUDED during training, see [Sina's ACL2019 paper](https://www.aclweb.org/anthology/P19-1063)
* `embeddings_rc_0split.npz`: this is a special split, called the listener in the ACL paper, this model is trained on the ENTIRE RefCoco data (including dev and test)
* `mscocolabels.md`: ids and names of mscoco labels, this is useful if you want to know which objects have been excluded when training the different embeddings

* `embeddings_coco_captioning.npz`: word embeddings taken from a captioning model trained on MSCOCO captions, see [this model](https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning)
