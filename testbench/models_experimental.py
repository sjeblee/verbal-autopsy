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
    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=5, dropout=0.0, ensemble=False, hidden_size=100):
        super(CNNText, self).__init__()
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

    def __init__(self, vocab_size, embed_size, embed_matrix, output_num, batch_size=10, hidden_size=100, pad_idx=0, num_layers=1):
        super(BaseRNN, self).__init__()

        embedding_tensor = torch.tensor(embed_matrix, dtype=torch.float64)
        self.num_layers = num_layers
        self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
        # self.embedding.weight = embedding_tensor
        self.embedding.weight.requires_grad = False 
        self.hidden_size = hidden_size

        self.dropout_emb = nn.Dropout(p=0.0)

        self.gru = nn.GRU(input_size=embed_size, hidden_size=hidden_size)

        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True)

        # self.batch_normalization = nn.BatchNorm1d(hidden_size * 2)
        # self.fully_connected = nn.Linear(hidden_size * 2, output_num)
        # self.out_softmax = nn.Softmax(dim=0)
        self.fc_to_label = nn.Linear(hidden_size, output_num)
        # self.hidden_layer = self.init_hidden(batch_size, hidden_size)
        self.hidden = self.init_hidden(batch_size)

    def init_hidden(self, batch_size):
        # return (Variable(torch.zeros(1, batch_size, self.hidden_size)), Variable(torch.zeros(1, batch_size, self.hidden_size)))
        return Variable(torch.zeros((1,batch_size,self.hidden_size)))
        
    def forward(self, x):

        # x : (batch, word num in sentence, vector size)
        print('raw x:')
        print(x)
        x_embedded = self.embedding(x)
        x_embedded = x_embedded.permute(1, 0, 2)
        x_embed = x_embedded.float()
        print('embed shape:')
        print(x_embed.shape)

        # batch_size = 10

        # hidden = self.init_hidden(x_embed.shape[0])

        # print("embed input:")
        # print(x_embed)
        '''
        print('prior lstm')
        for name, param in self.lstm.state_dict().items():
            print(name, param)

        '''
        gru_out, self.hidden = self.gru(x_embed, self.hidden)
        '''
        print('post lstm')
        for name, param in self.lstm.state_dict().items():
            print(name, param)
        '''

        # print("hidden shape: ")
        # print(self.hidden[0].shape, self.hidden[1].shape)

        # print("lstm_out shape: ")
        # print(lstm_out.shape)

        print("hidden last step: ")
        print(self.hidden.shape)
        print(self.hidden[-1].shape)
        print(self.hidden[-1])

        # row_idx = torch.arange(0, x.size(0)).long()

        # bn_tensor = torch.mean(gru_out[row_idx, :, :], dim=1)

        # batch_norm = self.batch_normalization(bn_tensor)

        #fc_out = self.fc_to_label(lstm_out[-1])
        fc_set = self.fc_to_label(self.hidden[-1])
        print('fc set:')
        print(fc_set)
        # print('fc_set: ')
        # print(fc_set)

        # batch_idx = torch.arange(0, x.size(0)).long()
        # fc_mean = torch.mean(fc_set[batch_idx, :, :], dim=1)
        # print('fc_mean: ')
        # print(fc_mean) 
        soft_out = F.log_softmax(fc_set, dim=-1)
        print('soft out:')
        print(soft_out)

        return soft_out

    '''
    def init_hidden(self, batch_size):
        return nn.Variable(torch.zeros(self.num_layers, batch_size, self.hidden_size)) 
    '''

class SequenceCNN2D(nn.Module):
    def __init__(self, embed_size, vocab_size, output_size, seq_len, hidden_size=32, dropout=0.2, conv_filters=128):
        super(SequenceCNN2D, self).__init__()

        # embedding_tensor = torch.tensor(embed_matrix, dtype=torch.float64)
        # self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
        # self.embedding.weight = embedding_tensor
        # self.embedding.weight.requires_grad = False 

        # INPUT (N, C=1, seq_len, embed_size)
        CHANNELS = 1

        # self.conv_01 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(3, embed_size))
        # self.conv_02 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(4, embed_size))
        self.conv_03 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(5, embed_size))
        self.conv_04 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(6, embed_size))
        self.conv_05 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(7, embed_size))
        self.conv_06 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(8, embed_size))
        # self.conv_07 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(9, embed_size))
        # self.conv_08 = nn.Conv2d(CHANNELS, conv_filters, kernel_size=(10, embed_size))

        self.dense_01 = nn.Linear(conv_filters * 4, hidden_size)
        self.dense_out = nn.Linear(hidden_size, output_size) 

        self.batch_norm_conv = nn.BatchNorm1d(conv_filters)
        self.batch_norm_out = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def seq_conv2d(self, x_emb, conv_2d, bn_1d, dropout_1d):
        x = conv_2d(x_emb)
        x = x[:,:,:,-1] # drop last dimension, now (N, 128, seq_len)
        x = bn_1d(x) 
        x = dropout_1d(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))
        return x 

    def forward(self, x):
        x = x.unsqueeze(1) # (N, seq_len, embed) -> (N, 1, seq_len, embed) 
        x_embed = x # input is now just straight up (N, max_len, embed_size)

        conv_out_1 = self.seq_conv2d(x_embed, self.conv_03, self.batch_norm_conv, self.dropout)
        conv_out_2 = self.seq_conv2d(x_embed, self.conv_04, self.batch_norm_conv, self.dropout)
        conv_out_3 = self.seq_conv2d(x_embed, self.conv_05, self.batch_norm_conv, self.dropout)
        conv_out_4 = self.seq_conv2d(x_embed, self.conv_06, self.batch_norm_conv, self.dropout)
        # conv_out_5 = self.seq_conv2d(x_embed, self.conv_05, self.batch_norm_conv, self.dropout)
        # conv_out_6 = self.seq_conv2d(x_embed, self.conv_06, self.batch_norm_conv, self.dropout)
        # conv_out_7 = self.seq_conv2d(x_embed, self.conv_07, self.batch_norm_conv, self.dropout)
        # conv_out_8 = self.seq_conv2d(x_embed, self.conv_08, self.batch_norm_conv, self.dropout)

        cn = torch.cat((conv_out_1, conv_out_2, conv_out_3, conv_out_4), 1)
        x = self.dropout(cn)
        x = torch.squeeze(x, 2)

        dense_out = self.dense_01(x)
        x = F.relu(dense_out)
        x = self.batch_norm_out(x)
        x = self.dropout(x)

        out = self.dense_out(x)

        return out 


class SequenceCNN(nn.Module):
    # Yoon Kim's sequence CNN 

    def __init__(self, embed_size, vocab_size, output_size, seq_len, hidden_size=32, dropout=0.2, conv_filters=128):
        super(SequenceCNN, self).__init__()

        # embedding_tensor = torch.tensor(embed_matrix, dtype=torch.float64)
        # self.embedding = nn.Embedding.from_pretrained(embedding_tensor)
        # self.embedding.weight = embedding_tensor
        # self.embedding.weight.requires_grad = False 

        self.conv_01 = nn.Conv1d(seq_len, conv_filters, kernel_size=3)
        self.conv_02 = nn.Conv1d(seq_len, conv_filters, kernel_size=4)
        self.conv_03 = nn.Conv1d(seq_len, conv_filters, kernel_size=5)
        self.conv_04 = nn.Conv1d(seq_len, conv_filters, kernel_size=6)

        self.dense_01 = nn.Linear(conv_filters * 2, hidden_size)
        self.dense_out = nn.Linear(hidden_size, output_size) 

        self.batch_norm_conv = nn.BatchNorm1d(conv_filters)
        self.batch_norm_out = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def seq_conv(self, x_emb, conv_1d, bn_1d, dropout_1d):
        x = conv_1d(x_emb)
        x = bn_1d(x)
        x = dropout_1d(x)
        x = F.relu(x)
        x = F.max_pool1d(x, x.size(2))
        return x 

    def forward(self, x):

        x_embed = x # input is now just straight up (N, max_len, embed_size)

        # conv_out_1 = self.seq_conv(x_embed, self.conv_01, self.batch_norm_conv, self.dropout)
        # conv_out_2 = self.seq_conv(x_embed, self.conv_02, self.batch_norm_conv, self.dropout)
        conv_out_3 = self.seq_conv(x_embed, self.conv_03, self.batch_norm_conv, self.dropout)
        conv_out_4 = self.seq_conv(x_embed, self.conv_04, self.batch_norm_conv, self.dropout)

        cn = torch.cat((conv_out_3, conv_out_4), 1)
        x = self.dropout(cn)
        x = torch.squeeze(x, 2)

        dense_out = self.dense_01(x)
        x = F.relu(dense_out)
        x = self.batch_norm_out(x)
        x = self.dropout(x)

        out = self.dense_out(x)

        return out 
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

    def __init__(self, input_size, hidden_size, output_size, n_layers=2, bidirectional=False):
        super(TextRNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = int(bidirectional) + 1
        self.gru = nn.GRU(input_size, hidden_size, n_layers, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden = self._init_hidden(batch_size=1)

    def forward(self, input): # seq_lengths
        print("Size of RNN input: ", str(input.size()))
        # batch size set to 1. Take one data at a time.
        batch_size = 1

        # Make a hidden
        # hidden = self._init_hidden(batch_size)

        # No embedding necessary since it takes output from CNN
        embedded = input.view(1, batch_size, -1)
        # print('embedded: ')
        # print(embedded)

        output, self.hidden = self.gru(embedded, self.hidden)

        # Use hidden layer as an input to the final layer
        # print('hidden: {}'.format(str(hidden.shape)))
        # print(hidden[0])
        # print('output: {}'.format(str(output.shape)))
        # print(output[0])
        # fc_output = self.fc(hidden[-1])
        fc_output = F.log_softmax(self.fc(output[-1]), dim=1)
        return fc_output

    def _init_hidden(self, batch_size):
        hidden = Variable(torch.zeros(self.n_layers * self.n_directions,
                             batch_size, self.hidden_size))
        if use_cuda:
            return hidden.cuda()
        else:
            return hidden

# ## 




