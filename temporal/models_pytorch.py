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

numpy.set_printoptions(threshold=numpy.inf)
debug = False
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
                print('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch, num_epochs, i//batch_size, num_examples//batch_size, loss.data[0]))
            i = i+batch_size

            del batchX
            del batchY
    if use_cuda:
        torch.cuda.empty_cache()
    return net

def test_nn(net, testX, batch_size=100, return_seqs=False, use_prev_labels=False):
    # Test the Model
    print_every = 1000
    pred = []
    i = 0
    length = len(testX)#.shape[0]
    print("testX len:", str(len(testX)))
    while i < length:
        if i%print_every == 0:
            print("test batch", str(i))
        if (i+batch_size) > length:
            batch_size = length-i
        x_array = numpy.asarray(testX[i:i+batch_size]).astype('float')
        #print("test x_array:", str(x_array.shape))
        if use_cuda:
            sample_tensor = torch.cuda.FloatTensor(x_array)
        else:
            sample_tensor = torch.FloatTensor(x_array)
        samples = Variable(sample_tensor)
        if return_seqs and use_prev_labels:
            max_length = sample_tensor.size(1)
            output_dim = 5
            hidden = None
            outputs = torch.zeros(batch_size, max_length, output_dim)
            for seq_id in range(batch_size):
                prev_label = torch.zeros(output_dim)
                #hidden = None
                for timestep in range(max_length):
                    timestep_x = sample_tensor[seq_id, timestep]
                    #if debug and (seq_id == 0):
                    #    print("timestep_x:", str(timestep_x.type()), str(timestep_x.size()), "prev_label:", str(prev_label.type()), str(prev_label.size()))
                    if use_prev_labels:
                        concat_array = torch.cat([timestep_x, prev_label], dim=0)
                    else:
                        concat_array = timestep_x
                    timestep_input = Variable(concat_array).view(1, -1)
                    output, hidden = net(timestep_input, hidden)
                    out = output.data[0]
                    out_index = torch.max(out, 0)[1]
                    out_label = torch.zeros(output_dim)
                    out_label[out_index] = 1
                    # Save the label for the next iteration
                    prev_label = out_label
                    outputs[seq_id, timestep] = out_label
            predicted = outputs
        elif return_seqs:
            outputs = net(samples)
            print("test outputs:", str(outputs.size()))
            _, predicted = torch.max(outputs.data, -1)
        else:
            outputs = net(samples)
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
        self.gru = nn.GRU(input_size, hidden_size, bidirectional=True, dropout=dropout_p)
        self.linear = nn.Linear(hidden_size*2, output_size)
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, input):
        #hidden = self.initHidden(input.size()[1])
        output, hn = self.gru(input, None)
        ## from (1, N, hidden) to (N, hidden)
        #rearranged = hn.view(hn.size()[1], hn.size(2))
        out1 = self.softmax(self.linear(output))
        return out1

    def initHidden(self, N):
        return Variable(torch.randn(1, N, self.hidden_size))

# GRU for sequences with label feeding
class GRU_seq(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, dropout_p=0.1):
        super(GRU_seq, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRUCell(input_size+output_size, hidden_size)
        self.linear = nn.Linear(hidden_size, output_size)
        self.softmax = F.log_softmax
        self.dp = dropout_p

    def forward(self, input, hidden):
        if hidden is None:
            hidden = self.initHidden()
        #print("input:", str(input.size()), "hidden:", str(hidden.size()))
        hn = self.gru(input, hidden)
        out0 = F.dropout(hn, p=self.dp)
        out1 = self.linear(out0)
        return out1, hn

    def initHidden(self):
        tens = torch.randn(1, self.hidden_size)
        if use_cuda:
            tens = tens.cuda()
        return Variable(tens)


''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_model(X, Y, num_nodes, activation='relu', modelname='gru', dropout=0.1, num_epochs=10, batch_size=1, loss_function='crossentropy', use_prev_labels=False):
    # Parameters
    hidden_size = num_nodes
    learning_rate = 0.001
    print_every = 1000
    teacher_forcing_ratio = 0.6
    class_weights = torch.FloatTensor([1.0, 1.0, 1.0, 1.0, 1.0])

    print("model:", modelname, "nodes:", str(num_nodes), "dropout:", str(dropout), "epochs:", str(num_epochs))

    if batch_size > 1:
        if type(X) is list and batch_size > 1:
            X = numpy.asarray(X)
        if type(Y) is list and batch_size > 1:
            Y = numpy.asarray(Y)
        X = X.astype('float')
        Y = Y.astype('float')

        num_examples = X.shape[0]
        input_dim = X.shape[-1]
        max_length = Y.shape[1]
        output_dim = Y.shape[-1]
        print("X:", str(X.shape), "Y:", str(Y.shape))
        print("max_length: ", str(max_length))
    else: # Leave X and Y as lists
        num_examples = len(X)
        input_dim = len(X[0][0])
        output_dim = len(Y[0][0])
        print("X list:", str(len(X)), "Y list:", str(len(Y)))

    print("input_dim: ", str(input_dim))
    print("output_dim: ", str(output_dim))

    if use_prev_labels:
        rnn = GRU_seq(input_dim, hidden_size, output_dim, dropout_p=dropout)
    else:
        rnn = GRU(input_dim, hidden_size, output_dim, dropout_p=dropout)

    if use_cuda:
        rnn = rnn.cuda()

    start = time.time()
    plot_losses = []

    optimizer = optim.Adam(rnn.parameters(), lr=learning_rate)

    if loss_function == 'cosine':
        criterion = nn.CosineEmbeddingLoss()
    elif loss_function == 'crossentropy':
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    elif loss_function == 'mse':
        criterion = nn.MSELoss()
    else:
        print("WARNING: need to add loss function!")

    # Train the model
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        while (i+batch_size) < num_examples:
            if i%print_every == 0:
                print("batch i=", str(i))
            # Make sure the data is in the proper numpy array format
            if batch_size == 1:
                batchXnp = X[i]
                batchYnp = Y[i]
                if type(batchXnp) is list:
                    batchXnp = numpy.asarray(batchXnp)
                if type(batchYnp) is list:
                    batchYnp = numpy.asarray(batchYnp)
            else:
                batchXnp = X[i:i+batch_size]
                batchYnp = Y[i:i+batch_size]

            # Convert to tensors
            batchXnp = batchXnp.astype('float')
            batchYnp = batchYnp.astype('float')
            if use_cuda:
                batchX = torch.cuda.FloatTensor(batchXnp)
                batchY = torch.cuda.FloatTensor(batchYnp)
            else:
                batchX = torch.FloatTensor(batchXnp)
                batchY = torch.FloatTensor(batchYnp)

            samples = Variable(batchX).view(batch_size, -1, input_dim)
            labels = Variable(batchY).view(batch_size, -1, output_dim)
            max_length = samples.size(1)
            if debug: print("batchX:", str(batchX.size()), "batchY:", str(batchY.size()))

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            loss = 0
            if use_prev_labels: # or loss_function == 'crossentropy'
                hidden = None
                for seq_id in range(batch_size):
                    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False
                    prev_label = torch.zeros(output_dim)
                    outputs = torch.zeros(max_length, output_dim)
                    true_labels = torch.zeros(max_length).long()
                    if use_cuda:
                        prev_label = prev_label.cuda()
                        outputs = outputs.cuda()
                        true_labels = true_labels.cuda()
                    if debug and seq_id == 0:
                        print("outputs size:", str(outputs.size()))
                    for timestep in range(max_length):
                        timestep_x = batchX[seq_id, timestep]
                        if debug and (seq_id == 0) and timestep == 0:
                            print("timestep_x:", str(timestep_x.type()), str(timestep_x.size()), "prev_label:", str(prev_label.type()), str(prev_label.size()))
                        if use_prev_labels:
                            concat_array = torch.cat([timestep_x, prev_label], dim=0)
                        else:
                            concat_array = timestep_x
                        timestep_input = Variable(concat_array).view(1, -1)
                        output, hidden = rnn(timestep_input, hidden)
                        out = output.data[0]
                        out_index = torch.max(out, 0)[1]
                        out_label = torch.zeros(output_dim)
                        if use_cuda:
                            out_label = out_label.cuda()
                        out_label[out_index] = 1
                        true_label = batchY[seq_id, timestep]
                        # Save the label for the next iteration
                        if use_teacher_forcing:
                            prev_label = true_label
                        else: # Use RNN predictions
                            prev_label = out_label
                        outputs[timestep] = out
                        true_index = torch.max(true_label, 0)[1].long()
                        #print("true_index", str(true_index.size()))
                        true_labels[timestep] = true_index[0]

                        # Compute loss - per label
                        #output_var = Variable(out, requires_grad=True).view(1, -1)
                        #if loss_function == 'crossentropy':
                        #    true_var = Variable(out_index).view(1)
                        #else:
                        #    true_var = Variable(batchY[seq_id, timestep]).view(1, -1)
                        #print('output:', str(output_var.size()), 'true:', true_var.size())
                        #loss += criterion(output_var, true_var)
                    # Computer loss - per sequence
                    outputs = Variable(outputs, requires_grad=True)
                    true_var = Variable(true_labels)
                    loss = criterion(outputs, true_var)
                    loss.backward()
                    optimizer.step()
            else:
                outputs = rnn(samples).view(max_length, -1)
                if loss_function == 'crossentropy':
                    for b in range(batch_size):
                        true_labels = torch.zeros(max_length).long()
                        for y in range(len(labels[b])):
                            true_label = labels[b][y].data
                            #print("true_label:", str(true_label.size()))
                            true_index = torch.max(true_label, 0)[1].long()
                            #print("true_index", str(true_index.size()))
                            true_labels[y] = true_index[0]
                        true_var = Variable(true_labels)
                        print("outputs:", str(outputs.size()))
                        print("true_var:", str(true_var.size()))
                        loss = criterion(outputs, true_var)
                        loss.backward()
                        optimizer.step()
                else:
                    true_var = labels
                    loss = criterion(outputs, true_var)
                    loss.backward()
                    optimizer.step()

            if (i) % print_every == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch, num_epochs, i//batch_size, num_examples//batch_size, loss.data[0]))
            i = i+batch_size

            del batchX
            del batchY
    return rnn

def test_rnn(rnn, testX, batch_size=100):
    return test_nn(rnn, testX, batch_size=batch_size, return_seqs=True)


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
