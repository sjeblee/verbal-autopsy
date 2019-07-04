###################################################################################################
# VERBAL-AUTOPSY DEVANAGARI CLASSIFIER TORCH MODEL LIBRARY (EXPERIMENTAL)
# =================================================================================================
#
# author: T. Ash Kumar
#
#
#                                                               
###################################################################################################

from __future__ import print_function

import math
import numpy
import os
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import data_util_experimental
import pickle

numpy.set_printoptions(threshold=numpy.inf)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)

use_cuda = False


###################################################################################################
# 
# MODEL COLLECTION
#
#       MODEL 1: 2D Temporal CNN
#       MODEL 2: Attention Transformer 
#       MODEL 3:
#       MODEL 4:
#
###################################################################################################

### CNNText

class SingleNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(SingleNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class CNNText(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, ensemble=False, hidden_size=100):
        super(CNN_Text, self).__init__()
        D = embed_dim
        C = class_num
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.ensemble = ensemble

        self.conv11 = nn.Conv2d(Ci, self.Co, (1, D))
        self.conv12 = nn.Conv2d(Ci, self.Co, (2, D))
        self.conv13 = nn.Conv2d(Ci, self.Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, self.Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, self.Co, (5, D))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.Co*self.Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)]
        x1 = self.conv_and_pool(x, self.conv11) # (N,Co)
        x2 = self.conv_and_pool(x, self.conv12) # (N,Co)
        x3 = self.conv_and_pool(x, self.conv13) # (N,Co)
        x4 = self.conv_and_pool(x, self.conv14) # (N,Co)
        x5 = self.conv_and_pool(x, self.conv15) # (N,Co)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        if not self.ensemble: # Train CNN with no ensemble
            logit = self.fc1(x)  # (N, C)
        else: # Train CNN with ensemble. Output of CNN will be input of another model
            logit = x
        return logit

class RNNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=False):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def _init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        else:
            return hidden

    def forward(self, input, seq_lengths):
        print("Size of RNN input: ", str(input.size()))
        # batch size set to 1. Take one data at a time.
        batch_size = 1

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # No embedding necessary since it takes output from CNN
        embedded = input.view(1, batch_size, -1)

        output, hidden = self.gru(embedded, hidden)

        # Use hidden layer as an input to the final layer
        #fc_output = self.fc(hidden[-1])
        fc_output = F.log_softmax(self.fc(output[0]), dim=1)
        return fc_output


### RNN Basic

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #print "embedding init size: " + str(input_size)
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        #print "forward: input: " + str(input.size())
        #embedded1 = self.embedding(input)
        embedded1 = input
        #print "embedded1: " + str(embedded1.size())
        embedded = embedded1.view(1, 1, -1)
        #print "embedded: " + str(embedded.size())
        #output = input
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #print "forward: input: " + type(input)
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

### Transformer (https://towardsdatascience.com/how-to-code-the-transformer-in-pytorch-24db27c8f9ec)
# later 

class Embedder(nn.Module):
    def __init__(self, vocab_size, dec_model):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dec_model)
    def forward(self, x):
        return self.embed(x)

class PosEncoder(nn.Module):
    def __init__(self, dec_model, max_seq_len = 80):

        super().__init__()
        self.dec_model = dec_model

        pe = torch.zeros(max_seq_len, dec_model) # constant position encoder matrix

        # fill in 
        for pos in range(max_seq_len):
            for i in range(0, dec_model, 2): # START 0, stop at dec_model, step by 2

                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/dec_model)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # enlargement of x relatively decreases size of pos-enc
        x = x * math.sqrt(self.dec_model) # enlarge / scale up by (dec_model)^{1/2}
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len], requires_grad=False).cuda() # add pos-enc constant
        return x

### Temporal Convolutional Network
# later

### Dilation Network
# later

### 1D CNN 
# later


##################################################

# https://machinelearningmastery.com/sequence-classification-lstm-recurrent-neural-networks-python-keras/
# sequence classification
# problem dev: variable-length movie review -> classify to sentiment

# sentences are a sequence of integers
# keras converts [3 5 3 2 4 2 1 1 6 3 ... ] into word embedding via Embedding layer
# 

# loading the dataset
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)
# over here, each X is a set of [3 4 3 2 2 1 4 2 ...]
# restricted only to top 5000 words
# we need to pad it next 

# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
# this allows the padding

# SIMPLE MODEL:
# Embedding layer => Dropout => LSTM layer with 100 memory units => Dropout => Dense output layer

# CNN + LSTM:
# Embedding => Conv1D => Max Pooling => LSTM => Dropout => Dense 

# https://www.analyticsvidhya.com/blog/2019/01/guide-pytorch-neural-networks-case-studies/
# SENTIMENT TEXT CLASSIFICATION (again)

# seeds
# np.random.seed(123)
# torch.manual_seed(123)
# torch.cuda.manual_seed(123)
# torch.backends.cudnn.deterministic = True

# from keras.preprocessing import text, sequence
# can also use torchtext

# ## create tokens 
# tokenizer = Tokenizer(num_words = MAX_WORDS)
# tokenizer.fit_on_texts(x_train)
# word_index = tokenizer.word_index

# ## convert texts to padded sequences 
# x_train = tokenizer.texts_to_sequences(x_train)
# x_train = pad_sequences(x_train, maxlen = MAX_LEN)

# ## CREATE DICTIONARY OF EMBEDDINGS
# embedding_matrix = np.zeros((len(word_index) + 1, EMB_VECTOR_LENGTH))
# for word, i in word_index.items():
#   emb_vector = model(word)
#   if emb_vector is not None:
#       embedding_matrix[i] = emb_vector

# ##

# https://github.com/keishinkickback/Pytorch-RNN-text-classification/blob/master/model.py

class BaseRNN(nn.Module):

    def __init__(self, vocab_size, embed_size, embed_matrix, output_num, hidden_size=100, pad_idx=0, num_layers=2):
        super(BaseRNN, self).__init__()

        embedding_tensor = torch.tensor(embed_matrix, dtype=torch.float64)
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
        # self.embedding.weight = embedding_tensor
        self.embedding.weight.requires_grad = False 

        self.dropout_emb = nn.Dropout(p=0.5)

        self.gru = nn.GRU(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.5,
            bidirectional=True
        )

        self.batch_normalization = nn.BatchNorm1d(hidden_size * 2)
        self.fully_connected = nn.Linear(hidden_size * 2, output_num)
        self.out_softmax = nn.Softmax(dim=0)
        
    def forward(self, x):

        # x : (batch, word num in sentence, vector size)
        x_embedded = self.embedding(x)
        x_embedded = self.dropout_emb(x_embedded)
        x_embed = x_embedded.float()
        
        gru_out, _ = self.gru(x_embed, None)

        row_idx = torch.arange(0, x.size(0)).long()

        bn_tensor = torch.mean(gru_out[row_idx, :, :], dim=1)

        batch_norm = self.batch_normalization(bn_tensor)

        fc_out = self.fully_connected(batch_norm)
        soft_out = self.out_softmax(fc_out)

        return soft_out

    def init_hidden(self, batch_size):
        return nn.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)) 

# ##

class TextCNN(nn.Module):

    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, ensemble=False, hidden_size=100):
        super(TextCNN, self).__init__()
        D = embed_dim
        C = class_num
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.ensemble = ensemble

        self.conv11 = nn.Conv2d(Ci, self.Co, (1, D))
        self.conv12 = nn.Conv2d(Ci, self.Co, (2, D))
        self.conv13 = nn.Conv2d(Ci, self.Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, self.Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, self.Co, (5, D))
        #self.conv16 = nn.Conv2d(Ci, Co, (6, D))
        #self.conv17 = nn.Conv2d(Ci, Co, (7, D))
        #self.conv18 = nn.Conv2d(Ci, Co, (8, D))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(self.Co*self.Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)]
        x1 = self.conv_and_pool(x, self.conv11) # (N,Co)
        x2 = self.conv_and_pool(x, self.conv12) # (N,Co)
        x3 = self.conv_and_pool(x, self.conv13) # (N,Co)
        x4 = self.conv_and_pool(x, self.conv14) # (N,Co)
        x5 = self.conv_and_pool(x, self.conv15) # (N,Co)
        #x6 = self.conv_and_pool(x,self.conv16) #(N,Co)
        #x7 = self.conv_and_pool(x,self.conv17)
        #x8 = self.conv_and_pool(x,self.conv18)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        if not self.ensemble: # Train CNN with no ensemble
            logit = self.fc1(x)  # (N, C)
        else: # Train CNN with ensemble. Output of CNN will be input of another model
            logit = x
        return logit

class TextRNNClassifier(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=False):
        super(TextRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, seq_lengths):
        print("Size of RNN input: ", str(input.size()))
        # batch size set to 1. Take one data at a time.
        batch_size = 1

        # Make a hidden
        hidden = self._init_hidden(batch_size)

        # No embedding necessary since it takes output from CNN
        embedded = input.view(1, batch_size, -1)

        output, hidden = self.gru(embedded, hidden)

        # Use hidden layer as an input to the final layer
        #fc_output = self.fc(hidden[-1])
        fc_output = F.log_softmax(self.fc(output[0]), dim=1)
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        else:
            return hidden

# ## 




