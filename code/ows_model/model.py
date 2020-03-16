import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable

from utils.build_vocab import Vocabulary
import pickle

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(EncoderCNN, self).__init__()
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.linear = nn.Linear(resnet.fc.in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)
        self.init_weights()

    def init_weights(self):
        """Initialize the weights."""
        self.linear.weight.data.normal_(0.0, 0.02)
        self.linear.bias.data.fill_(0)

    def forward(self, images):
        """Extract the image feature vectors."""
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features


# taken and adapted from:
#https://github.com/Andras7/word2vec-pytorch/tree/master/word2vec
class ObjectWordNet(nn.Module):


    def __init__(self, emb_size, emb_dimension):
        super(SkipGramModel, self).__init__()
        self.emb_size = emb_size
        self.emb_dimension = emb_dimension
        self.u_embeddings = nn.Embedding(emb_size, emb_dimension, sparse=True)

        initrange = 1.0 / self.emb_dimension
        init.uniform_(self.u_embeddings.weight.data, -initrange, initrange)

    def forward(self, words, pos_features, neg_features):
        embedding = self.u_embeddings(words)

        score = torch.sum(torch.mul(pos_features, embedding), dim=1)
        score = torch.clamp(score, max=10, min=-10)
        score = -F.logsigmoid(score)

        neg_score = torch.bmm(neg_features, embedding.unsqueeze(2)).squeeze()
        #neg_score = torch.sum(torch.mul(neg_features, embedding), dim=1)
        neg_score = torch.clamp(neg_score, max=10, min=-10)
        neg_score = -torch.sum(F.logsigmoid(-neg_score), dim=1)

        return torch.mean(score + neg_score)

    def save_embedding(self, id2word, file_name):
        embedding = self.u_embeddings.weight.cpu().data.numpy()
        with open(file_name, 'w') as f:
            f.write('%d %d\n' % (len(id2word), self.emb_dimension))
            for wid, w in id2word.items():
                e = ' '.join(map(lambda x: str(x), embedding[wid]))
                f.write('%s %s\n' % (w, e))




    # def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
    #     """Set the hyper-parameters and build the layers."""
    #     super(DecoderRNN, self).__init__()
    #     self.embed = nn.Embedding(vocab_size, embed_size)
    #     self.merge = nn.Linear(embed_size+feat_size, hidden_size)
    #     self.relu = nn.ReLU()
    #     self.linear = nn.Linear(hidden_size, 1)
    #     self.init_weights()
    #
    # def init_weights(self):
    #     """Initialize weights."""
    #     self.embed.weight.data.uniform_(-0.1, 0.1)
    #     self.merge.weight.data.uniform_(-0.1, 0.1)
    #     self.merge.bias.data.fill_(0)
    #     self.linear.weight.data.uniform_(-0.1, 0.1)
    #     self.linear.bias.data.fill_(0)
    #
    # def forward(self, features, words, lengths):
    #     """Decode image feature vectors and generates captions."""
    #     embeddings = self.embed(words)
    #     embeddings = torch.cat((features, embeddings), 1)
    #     embeddings = self.merge(embeddings)
    #     embeddings = self.relu(embeddings)
    #     outputs = self.linear(embeddings)
    #     return outputs
