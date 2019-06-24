#!/usr/bin/python3
# -*- coding: utf-8 -*-
# PyTorch models for keyword classification
# @author sjeblee@cs.toronto.edu

import numpy
import time
import torch
import torch.nn as nn

numpy.set_printoptions(threshold=numpy.inf)

# Use GPU is available, otherwise use CPU
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:0')

use_cuda = False

class LinearNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_epochs=10, dropout_p=0.1):
        super(LinearNN, self).__init__()
        self.hidden_size = hidden_size
        self.epochs = num_epochs
        self.dropout = dropout_p
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
    def fit(self, X, Y):
        st = time.time()
        #Xarray = numpy.asarray(X).astype('float')
        Xarray = X.astype('float')
        Yarray = Y.astype('int')
        print('X numpy shape:', str(Xarray.shape), 'Y numpy shape:', str(Yarray.shape))

        X_len = Xarray.shape[0]
        steps = 0
        #best_acc = 0
        #last_step = 0
        #log_interval = 1000
        batch_size = 100
        #num_batches = math.ceil(X_len/batch_size)
        learning_rate = 0.001

        if use_cuda:
            self = self.to(tdevice)

        # Train
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_function = nn.CrossEntropyLoss()

        steps = 0
        for epoch in range(self.epochs):
            print('epoch:', str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = Xarray[permutation]
            Yiter = Yarray[permutation]

            while i+batch_size < X_len:
                batchX = Xiter[i:i+batch_size]
                batchY = Yiter[i:i+batch_size]
                Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    Xtensor = Xtensor.cuda()
                    Ytensor = Ytensor.cuda()
                feature = Xtensor
                target = Ytensor
                i = i+batch_size

                optimizer.zero_grad()
                logit = self(feature)
                #loss = F.cross_entropy(logit, torch.max(target, 1)[1])
                loss = loss_function(logit, target)
                loss.backward()
                optimizer.step()

                steps+=1
            ct = time.time() - st
            unit = 's'
            if ct> 60:
                ct = ct/60
                unit = 'mins'
            print('Time so far:', str(ct), unit)
            print('Loss:', str(loss))

    def predict(self, testX, threshold=0.01):
        # Test the Model
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)
        testX = testX.astype('float')
        pred = [] # Original prediction without using threshold for ill-defined.
        #new_pred = [] # New prediction by assigning to UNKNOWN class if below the threshold
        #probs = [] # Probabilities, the softmax output of the final layer

        batch_size = 100
        i = 0
        length = len(testX)
        pred = []
        while i < length:
            if i % 10 == 0:
                print('test batch', str(i))
            if (i+batch_size) > length:
                batch_size = length-i
            if use_cuda:
                sample_tensor = torch.cuda.FloatTensor(testX[i:i+batch_size])
            else:
                sample_tensor = torch.FloatTensor(testX[i:i+batch_size])
            outputs = self(sample_tensor)
            print('outputs:', len(outputs))

            outputs_ls = logsoftmax(outputs)
            #outputs_softmax = softmax(outputs)
            predicted = torch.max(outputs_ls, 1)[1].data
            #probabilities = torch.max(outputs_softmax, 1)[0].data
            pred = pred + predicted.tolist()
            print(str(len(pred)))

            del sample_tensor
            i = i+batch_size

        return pred
