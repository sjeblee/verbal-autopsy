# @author sjeblee@cs.toronto.edu

from __future__ import print_function

import math
import numpy
import os
import pickle
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F


numpy.set_printoptions(threshold=numpy.inf)

# Use GPU is available, otherwise use CPU
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    tdevice = torch.device('cuda:0')

use_cuda = False


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
class CNN_Text(nn.Module):

    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.1, ensemble=False, num_epochs=10, loss_func='categorical_crossentropy'):
        super(CNN_Text, self).__init__()
        D = embed_dim
        C = class_num
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.ensemble = ensemble
        self.epochs = num_epochs
        self.loss_func = loss_func

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

    ''' Create and train a CNN model
        Hybrid features supported - pass structured feats as X2
        Does NOT support joint training yet
        returns: the CNN model
    '''
    def fit(self, X, Y, kernel_sizes=5, pretrainX=[], pretrainY=[], query_vectors=None):
        # Train the CNN, return the model
        st = time.time()
        use_query = (query_vectors is not None)

        # Check for pretraining
        pretrain = False
        if len(pretrainX) > 0 and len(pretrainY) > 0:
            pretrain = True
            print("Using pretraining")

        # Params
        Xarray = numpy.asarray(X).astype('float')
        dim = Xarray.shape[-1]
        num_labels = Y.shape[-1]
        batch_size = 32
        learning_rate = 0.001

        #num_epochs = num_epochs
        #best_acc = 0
        #last_step = 0
        #log_interval = 1000

        if use_cuda:
            self = self.to(tdevice)
        if use_cuda and use_query:
            self.query_vectors = self.query_vectors.cuda()

        if pretrain:
            print("pretraining...")
            for k in range(len(pretrainX)):
                trainX = pretrainX[k]
                trainY = pretrainY[k]
                pre_labels = trainY.shape[-1]

                if k > 0: # Replace the last layer
                    self.fc1 = nn.Linear(self.Co*self.Ks, pre_labels)

                # Pre-train the model
                self.train(trainX, trainY, learning_rate, batch_size)

            # Replace the last layer for the final model
            self.fc1 = nn.Linear(self.Co*self.Ks, num_labels)
            ct = time.time() - st
            unit = 's'
            if ct > 60:
                ct = ct/60
                unit = 'm'
            print('CNN pretraining took: ', str(ct), unit)

        '''
        else: # No pre-training
            if use_query:
                dropout = 0.1
                cnn = CNN_Query(query_vectors, dim, num_labels, dropout=dropout, kernel_sizes=kernel_sizes)
            else:
                cnn = CNN_Text(dim, num_labels, dropout=dropout, kernel_sizes=kernel_sizes)
        '''

        # Train final model
        self.train(X, Y, learning_rate, batch_size)

    def train(self, X, Y, learning_rate, batch_size, query=False):
        Xarray = numpy.asarray(X).astype('float')
        Yarray = Y.astype('int')
        X_len = Xarray.shape[0]
        #dim = Xarray.shape[-1]
        #num_labels = Yarray.shape[-1]
        #num_batches = math.ceil(X_len/batch_size)
        print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))
        steps = 0
        st = time.time()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(self.epochs):
            print("epoch", str(epoch))
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

                optimizer.zero_grad()
                if query:
                    logit, attn_maps = self(Xtensor)
                else:
                    logit = self(Xtensor)

                loss = F.cross_entropy(logit, torch.max(Ytensor, 1)[1])
                print('loss: ', str(loss.data[0]))
                loss.backward()
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

    def predict(self, testX, testids=None, labelencoder=None, collapse=False, threshold=0.1, probfile=None):
        y_pred = [] # Original prediction if threshold is not in used for ill-defined.
        y_pred_softmax = []
        y_pred_logsoftmax = []
        softmax = nn.Softmax(dim=1)
        logsoftmax = nn.LogSoftmax(dim=1)

        new_y_pred = [] # class prediction if threshold for ill-difined is used.
        if probfile is not None:
            probs = []

        for x in range(len(testX)):
            input_row = testX[x]

            icd = None
            if icd is None:
                input_tensor = torch.from_numpy(input_row.astype('float')).float().unsqueeze(0)
                if use_cuda:
                    #input_tensor = input_tensor.contiguous().cuda()
                    input_tensor = input_tensor.cuda()
                print('input_tensor:', input_tensor.size())
                print('input_tensor[0, 0]:', input_tensor[0, 0])
                #icd_var, attn_maps = model(Variable(input_tensor))
                icd_var = self(input_tensor)
                # Softmax and log softmax values
                icd_vec = logsoftmax(icd_var)
                icd_vec_softmax = softmax(icd_var)
                icd_code = torch.max(icd_vec, 1)[1].data[0]

                # Save the first example attn map
                '''
                if x == 0:
                    tempfile = '/u/sjeblee/research/va/data/cnn_query_cghr/attn_0.csv'
                    outf = open(tempfile, 'w')
                    for row in attn_maps[0]:
                        row = row.squeeze()
                        for i in range(row.size(0)):
                            outf.write(str(row.data[i]) + ',')
                        outf.write('\n')
                    outf.close()
                '''

    	    # Assign to ill-defined class if the maximum probabilities is less than a threshold.
    	    #icd_prob = torch.max(icd_vec_softmax,1)[0]
    	    #if icd_prob < threshold:
                #    new_y_pred.append(9)
    	    #else:
    	    #    new_y_pred.append(icd_code)

                # Save the probabilties
                #if probfile is not None:
                #    probs.append(icd_prob)

            y_pred.append(icd_code)

        if probfile is not None:
            pickle.dump(probs, open(probfile, "wb"))
        #print "Probabilities: " + str(probs)

        return y_pred  # Uncomment this line if threshold is not in used.
        #return new_y_pred  # Comment this line out if threshold is not in used.
