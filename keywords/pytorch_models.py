#!/usr/bin/python3
# -*- coding: utf-8 -*-
# PyTorch models for keyword classification
# @author sjeblee@cs.toronto.edu

import numpy
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids

numpy.set_printoptions(threshold=numpy.inf)

# Use GPU is available, otherwise use CPU
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:2')

#use_cuda = False

class LinearNN(nn.Module):

    def __init__(self, input_size, hidden_size, num_classes, num_epochs=10, dropout_p=0.1):
        super(LinearNN, self).__init__()
        self.hidden_size = hidden_size
        self.epochs = num_epochs
        self.dropout = dropout_p
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.logsoftmax = nn.LogSoftmax(dim=1)

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
        #softmax = nn.Softmax(dim=1)
        #logsoftmax = nn.LogSoftmax(dim=1)
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

            outputs_ls = self.logsoftmax(outputs)
            #outputs_softmax = softmax(outputs)
            predicted = torch.max(outputs_ls, 1)[1].data
            #probabilities = torch.max(outputs_softmax, 1)[0].data
            pred = pred + predicted.tolist()
            print(str(len(pred)))

            del sample_tensor
            i = i+batch_size

        return pred


######################################################################
# Convolutional Neural Network
# ----------------------------
#
# 5 convlutional layers with max pooling, followed by a fully connected network
# Arguments:
#	embed_dim	: dimension of a word vector
#	class_num	: number of classes
#	kernel_num	: number of channels produced by the convolutions
#	kernel_sizes	: size of convolving kernels
#	dropout		: dropout to prevent overfitting
#	ensemble	: if true, used as input of RNNClassifier
#			  if false, used independently and make prediction
#	hidden_size	: number of nodes in hidden layers
#
#
class ElmoCNN(nn.Module):

    def __init__(self, input_size, num_classes, num_epochs=10, dropout_p=0.1, kernel_num=100, kernel_sizes=3, loss_func='crossentropy'):
        super(ElmoCNN, self).__init__()
        options_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_options.json"
        weight_file = "/u/sjeblee/research/data/elmo/weights/elmo_2x4096_512_2048cnn_2xhighway_weights_PubMed_only.hdf5"
        self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)

        D = input_size
        C = num_classes
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.epochs = num_epochs
        self.loss_func = loss_func

        self.conv11 = nn.Conv2d(Ci, self.Co, (1, D))
        self.conv12 = nn.Conv2d(Ci, self.Co, (2, D))
        self.conv13 = nn.Conv2d(Ci, self.Co, (3, D))
        #self.conv14 = nn.Conv2d(Ci, self.Co, (4, D))
        #self.conv15 = nn.Conv2d(Ci, self.Co, (5, D))
        #self.conv16 = nn.Conv2d(Ci, Co, (6, D))
        #self.conv17 = nn.Conv2d(Ci, Co, (7, D))
        #self.conv18 = nn.Conv2d(Ci, Co, (8, D))

        self.dropout = nn.Dropout(dropout_p)
        self.fc1 = nn.Linear(self.Co*self.Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        batch_size = len(x)
        character_ids = batch_to_ids(x).to(tdevice)
        embeddings = self.elmo(character_ids)['elmo_representations']
        #print('elmo embeddings:', embeddings[0].size())
        X = embeddings[0].view(batch_size, -1, 1024) # (N, W, D)

        # Pad to 10 words
        if X.size(1) > 10:
            X = X[:, 0:10, :]
        elif X.size(1) < 10:
            pad_size = 10 - X.size(1)
            zero_vec = torch.zeros(X.size(0), pad_size, X.size(2), device=tdevice)
            X = torch.cat((X, zero_vec), dim=1)

        x = X.unsqueeze(1)  # (N, Ci, W, D)]
        #print('x size:', x.size())
        x1 = self.conv_and_pool(x, self.conv11) # (N,Co)
        x2 = self.conv_and_pool(x, self.conv12) # (N,Co)
        x3 = self.conv_and_pool(x, self.conv13) # (N,Co)
        #x4 = self.conv_and_pool(x, self.conv14) # (N,Co)
        #x5 = self.conv_and_pool(x, self.conv15) # (N,Co)
        #x6 = self.conv_and_pool(x,self.conv16) #(N,Co)
        #x7 = self.conv_and_pool(x,self.conv17)
        #x8 = self.conv_and_pool(x,self.conv18)
        x = torch.cat((x1, x2, x3), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)

        logit = self.fc1(x)  # (N, C)
        return logit

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, X, Y):
        # Train the CNN, return the model
        #st = time.time()

        # Params
        #Xarray = numpy.asarray(X).astype('float')
        #dim = Xarray.shape[-1]
        #num_labels = Y.shape[-1]
        batch_size = 16
        learning_rate = 0.001

        if use_cuda:
            self = self.to(tdevice)

        # Train final model
        self.train(X, Y, learning_rate, batch_size)

    def train(self, X, Y, learning_rate, batch_size):
        #Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = len(X)
        print('X len:', X_len)
        print('Y numpy shape:', str(Yarray.shape))
        steps = 0
        st = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        if self.loss_func == 'crossentropy':
            loss = nn.CrossEntropyLoss()
        else:
            print('ERROR: unrecognized loss function name')

        for epoch in range(self.epochs):
            print('epoch', str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            perm = torch.from_numpy(numpy.random.permutation(X_len))
            permutation = perm.long()
            perm_list = perm.tolist()
            Xiter = [X[i] for i in perm_list]
            #Xiter = X[permutation]
            Yiter = Yarray[permutation]

            while i+batch_size < X_len:
                batchX = Xiter[i:i+batch_size]
                batchY = Yiter[i:i+batch_size]
                #Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    #Xtensor = Xtensor.cuda()
                    Ytensor = Ytensor.to(tdevice)

                optimizer.zero_grad()
                logit = self(batchX)

                loss_val = loss(logit, Ytensor)
                #print('loss: ', loss_val.data.item())
                loss_val.backward()
                optimizer.step()
                steps += 1
                i = i+batch_size

            # Print epoch time
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)
            print('loss: ', loss_val.data.item())

    def predict(self, testX, testids=None, labelencoder=None, collapse=False, threshold=0.1, probfile=None):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.
        #y_pred_softmax = []
        #y_pred_logsoftmax = []
        #softmax = nn.Softmax(dim=1)
        new_y_pred = [] # class prediction if threshold for ill-difined is used.
        #if probfile is not None:
        #    probs = []

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                #input_tensor = torch.from_numpy(input_row.astype('float')).float().unsqueeze(0)
                #if use_cuda:
                    #input_tensor = input_tensor.cuda()
                #print('input_row:', len(input_row))
                #print('input_row[0]:', input_row[0])
                #icd_var, attn_maps = model(Variable(input_tensor))
                icd_var = self([input_row])
                # Softmax and log softmax values
                icd_vec = self.logsoftmax(icd_var).squeeze()
                #print('pred vector:', icd_vec.size(), icd_vec)
                print('argmax:', torch.argmax(icd_vec))
                #icd_vec_softmax = softmax(icd_var)
                cat = torch.argmax(icd_vec).item()
                if x == 0:
                    print('cat:', cat)
                #icd_code = cat

            y_pred.append(cat)

        #print "Probabilities: " + str(probs)

        return y_pred  # Uncomment this line if threshold is not in used.
        #return new_y_pred  # Comment this line out if threshold is not in used.
