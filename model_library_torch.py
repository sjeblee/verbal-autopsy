#!/usr/bin/python
# Neural network model functions in PyTorch
# @author sjeblee@cs.toronto.edu

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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import data_util

numpy.set_printoptions(threshold=numpy.inf)
use_cuda = torch.cuda.is_available()

# Models from PyTorch Tutorial:
######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        print "forward: input: " + str(type(input))
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#
######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

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
######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.

''' Create and train a feed-forward neural network model. Features selection should be done before passing to this function
    X: a python list or numpy array of training data of shape [num_samples, num_features]
    Y: a python list or numpy array of training labels of shape [num_samples]
    returns: the model and the modified X and Y arrays
'''
def nn_model(X, Y, num_nodes, act, num_epochs=10):
    X = numpy.asarray(X)
    Y = to_categorical(Y)
    #print "X.shape: " + str(X.shape)
    #print "Y.shape: " + str(Y.shape)

    #print "neural network: nodes: " + str(num_nodes)
    #nn = Sequential([Dense(num_nodes, input_dim=X.shape[-1]),
    #                Activation(act),
                    #Dense(num_nodes, input_dim=num_feats),
                    #Activation(activation),
   #                 Dense(Y.shape[1]),
    #                Activation('softmax'),])
        
    #nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #nn.fit(X, Y, epochs=num_epochs)
    #nn.summary()
    #return nn, X, Y

''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_model(X, Y, num_nodes, activation='sigmoid', modelname='lstm', dropout=0.1, X2=[], pretrainX=[], pretrainY=[], pretrainX2=[], initial_states=None, num_epochs=15):
    #Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    inputs = []
    input_arrays = [X]
    hybrid = False
    pretrain = False
    if len(pretrainX) > 0 and len(pretrainY) > 0:
        pretrain = True
        print "Using pretraining"

    print "model: " + modelname + " nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)

    print "NOT IMPLEMENTED"


''' Create and train a CNN model
    Hybrid features supported - pass structured feats as X2
    Does NOT support joint training yet
    returns: the CNN model
'''
def cnn_model(X, Y, act=None, windows=[1,2,3,4,5], X2=[], num_epochs=10, loss_func='categorical_crossentropy'):
    #Y = to_categorical(Y)
    X = numpy.asarray(X)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    print "train X shape: " + str(X.shape)
    print "max_seq_len: " + str(max_seq_len)
    inputs = []
    input_arrays = [X]
    hybrid = False

    print "NOT IMPLEMENTED"

def train_encoder_decoder(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length):
    # Parameters
    teacher_forcing_ratio = 0.5
    loss = 0

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    print "input_length: " + str(input_length) + ", target_length: " + str(target_length)
    input_dim = input_variable.size()[-1]
    output_dim = target_variable.size()[-1]
    start_token = torch.zeros(input_dim)
    end_token = torch.ones(output_dim).long()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    for ei in range(input_length):
        print "ei: " + str(ei)
        print "encoder_hidden: " + str(type(encoder_hidden.data))
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(start_token)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.FloatTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == end_token:
                break

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def encoder_decoder_model(X, Y, num_nodes, print_every=1000, plot_every=100, learning_rate=0.01, num_epochs=10, loss_function=nn.CosineEmbeddingLoss):
    print "Encoder-decoder model: " + str(num_nodes)
    n_iters = len(X)
    print "epochs: " + str(num_epochs) + " iters: " + str(n_iters)
    if use_cuda:
        X = torch.cuda.FloatTensor(X)
        Y = torch.cuda.FloatTensor(Y)
    else:
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    print "X: " + str(X.size())
    print "Y: " + str(Y.size())
    hidden_size = num_nodes
    max_length = Y.size()[1]
    output_dim = Y.size()[-1]
    encoder1 = EncoderRNN(X.size()[-1], hidden_size)
    decoder1 = AttnDecoderRNN(hidden_size, output_dim, dropout_p=0.1)


    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder1.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = zip(X, Y)
    criterion = loss_function

    for ep in range(num_epochs):
        print 'epoch ' + str(ep)
        for iter in range(1, n_iters + 1):
            training_pair = training_pairs[iter-1]
            input_variable = training_pair[0]
            target_variable = training_pair[1]

            loss = train_encoder_decoder(input_variable, target_variable, encoder1, decoder1, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

    return encoder1, decoder1, output_dim

'''
    testX: python list of input sequences (python lists)
    testY: python list of output sequences (also python lists)
    max_length: the maximum sequence length
'''
def test_encoder_decoder(encoder, decoder, testX, max_length, output_dim):
    predY = []
    for test_seq in testX:
        decoded_seq, decoded_attn = decode(encoder, decoder, test_seq, max_length, output_dim)
        pred_seq = decoded_seq.storage.tolist()
        predY.append(pred_seq)
    return predY

def decode(encoder, decoder, input_vec, max_length, output_dim):
    if use_cuda:
        input_variable = Variable(torch.cuda.FloatTensor(input_vec))
    else:
        input_variable = Variable(torch.FloatTensor(input_vec))
    input_length = input_variable.size()[0]
    input_dim = input_variable.size()[-1]
    start_token = torch.zeros(input_dim)
    end_token = torch.ones(output_dim)
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(start_token)  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoded_seq = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == end_token:
            decoded_seq.append(ni)
            break
        else:
            decoded_seq.append(ni)

        decoder_input = Variable(torch.FloatTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_seq, decoder_attentions[:di + 1]

##################################################
# HELPER FUNCTIONS
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def evaluateAndShowAttention(input_sentence, encoder, attn_decoder):
    output_words, attentions = evaluate(encoder, attn_decoder, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
