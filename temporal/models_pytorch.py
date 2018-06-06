#!/usr/bin/python3
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
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

import data_util3

numpy.set_printoptions(threshold=numpy.inf)
use_cuda = False #torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(2)

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

''' Create and train a feed-forward neural network model. Features selection should be done before passing to this function
    X: a python list or numpy array of training data of shape [num_samples, num_features]
    Y: a python list or numpy array of training labels of shape [num_samples]
    returns: the model and the modified X and Y arrays
'''
def nn_model(X, Y, num_nodes, act, num_epochs=10, batch_size=100):
    print("use_cuda", str(use_cuda))
    if type(X) is list:
        X = numpy.asarray(X)
    if type(Y) is list:
        Y = numpy.asarray(Y)
    print("X: ", str(X.shape))
    print("Y: ", str(Y.shape))

    # Hyper Parameters
    input_dim = X.shape[-1]
    num_examples = X.shape[0]
    num_classes = Y.shape[-1]
    learning_rate = 0.001

    print("neural network: nodes: ", str(num_nodes))
    net = Net(input_dim, num_nodes, num_classes)

    if use_cuda:
        net = net.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train the Model
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        while (i+batch_size) < num_examples:
            if i%10000 == 0:
                print("batch i=", str(i))
            # Convert torch tensor to Variable
            batchX = torch.Tensor(X[i:i+batch_size])
            batchY = torch.Tensor(Y[i:i+batch_size]).long()
            if use_cuda:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            #print("X: ", str(X.size()))
            #print("Y: ", str(Y.size()))
            samples = Variable(batchX)
            labels = Variable(batchY)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i) % 10000 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch, num_epochs, i//batch_size, num_examples//batch_size, loss.data[0]))
            i = i+batch_size

            del batchX
            del batchY
    if use_cuda:
        torch.cuda.empty_cache()
    return net

def test_nn(net, testX, batch_size=100, return_seqs=False):
    # Test the Model
    if type(testX) is list:
        testX = numpy.asarray(testX)
    testX = testX.astype('float')
    print_every = 1000
    pred = []
    i = 0
    length = len(testX)#.shape[0]
    print "testX len:" + str(len(testX))
    while i < length:
        if i%print_every == 0:
            print("test batch", str(i))
        if (i+batch_size) > length:
            batch_size = length-i
        if use_cuda:
            sample_tensor = torch.cuda.FloatTensor(testX[i:i+batch_size])
        else:
            sample_tensor = torch.FloatTensor(testX[i:i+batch_size])
        samples = Variable(sample_tensor)
        outputs = net(samples)
        if return_seqs:
            predicted = outputs.data
        else:
            _, predicted = torch.max(outputs.data, 1)
        pred = pred + predicted.tolist()
        del sample_tensor
        i = i+batch_size
    return pred

# GRU
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=False, dropout=dropout_p)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        #hidden = self.initHidden(input.size()[1])
        output, hn = self.gru(input, None)
        ## from (1, N, hidden) to (N, hidden)
        #rearranged = hn.view(hn.size()[1], hn.size(2))
        out1 = self.linear(output)
        return out1

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))


''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_model(X, Y, num_nodes, activation='sigmoid', modelname='gru', dropout=0.1, num_epochs=10, loss_function='crossentropy'):
    print("model:", modelname, "nodes:", str(num_nodes), "dropout:", str(dropout), "epochs:", str(num_epochs))

    if type(X) is list:
        X = numpy.asarray(X)
    if type(Y) is list:
        Y = numpy.asarray(Y)
    X = X.astype('float')
    Y = Y.astype('float')

    #n_iters = len(X)
    if use_cuda:
        print("using cuda")
        X = torch.cuda.FloatTensor(X)
        Y = torch.cuda.FloatTensor(Y)
    else:
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    print("X: ", str(X.size()))
    print("Y: ", str(Y.size()))
    hidden_size = num_nodes
    num_examples = X.size()[0]
    input_dim = X.size()[-1]
    max_length = Y.size()[1]
    output_dim = Y.size()[-1]
    learning_rate = 0.0001
    batch_size = 100
    print("input_dim: ", str(input_dim))
    print("max_length: ", str(max_length))
    print("output_dim: ", str(output_dim))
    rnn = GRU(input_dim, hidden_size, output_dim, dropout_p=dropout)

    if use_cuda:
        rnn = rnn.cuda()

    start = time.time()
    plot_losses = []
    print_every = 1000
    #print_loss_total = 0  # Reset every print_every
    #plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = zip(X, Y)
    if loss_function == 'cosine':
        criterion = nn.CosineEmbeddingLoss()
    elif loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    else:
        print("WARNING: need to add loss function!")

    # Train the Model
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        while (i+batch_size) < num_examples:
            if i%print_every == 0:
                print("batch i=", str(i))
            # Convert torch tensor to Variable
            batchX = torch.Tensor(X[i:i+batch_size])
            batchY = torch.Tensor(Y[i:i+batch_size])
            if use_cuda:
                batchX = batchX.cuda()
                batchY = batchY.cuda()
            #print("X: ", str(X.size()))
            #print("Y: ", str(Y.size()))
            samples = Variable(batchX)
            labels = Variable(batchY)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = rnn(samples)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i) % print_every == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch, num_epochs, i//batch_size, num_examples//batch_size, loss.data[0]))
            i = i+batch_size

            del batchX
            del batchY
    return rnn

def test_rnn(rnn, testX):
    return test_nn(rnn, testX, return_seqs=True)


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
    print("TODO")
    # Set up figure with colorbar
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(attentions.numpy(), cmap='bone')
    #fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #plt.show()

#def evaluateAndShowAttention(input_sentence, encoder, attn_decoder):
#    output_words, attentions = decode(encoder, attn_decoder, input_sentence, 10, 100)
#    print('input =', input_sentence)
#    print('output =', ' '.join(output_words))
#    showAttention(input_sentence, output_words, attentions)
