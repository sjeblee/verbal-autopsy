#!/usr/bin/python3

import math
import numpy
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from allennlp.modules.elmo import Elmo, batch_to_ids
from torch import optim
from keras.utils.np_utils import to_categorical
from torch.autograd import Variable
from word2vec import load
from word2vec import get
from utils import timeSince
numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    cuda = torch.device("cuda:0")

# GRU with GRU encoder, input: (conversations (1), utterances, words, embedding_dim)
class CNN_ELMO(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, hidden_size = 100, USE_SERVER=True):
          super(CNN_ELMO, self).__init__()
          options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
          if USE_SERVER == True:
              weight_file = '/u/yanzhaod/data/elmo_pubMed_only.hdf5'
          else:
              weight_file = 'D:/projects/zhaodong/research/elmo_pubMed_only.hdf5'
          self.hidden_size = hidden_size
          self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)
          
          D = embed_dim
          C = class_num
          Ci = 1
          Co = kernel_num
          Ks = kernel_sizes
          self.conv11 = nn.Conv2d(Ci, Co, (1, D))
          self.conv12 = nn.Conv2d(Ci, Co, (2, D))
          self.conv13 = nn.Conv2d(Ci, Co, (3, D))
          self.conv14 = nn.Conv2d(Ci, Co, (4, D))
          self.conv15 = nn.Conv2d(Ci, Co, (5, D))
          self.dropout = nn.Dropout(dropout)
          self.fc1 = nn.Linear(Co*Ks, C)
          if debug:
              print("embedding dimension: "+str(embed_dim))
              print("number of classes: "+str(class_num))
    def conv_and_pool(self, x, conv):
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
          return x
    ''' Input is a list of strings
    '''
    def forward(self, x):
         x = x.unsqueeze(1)  # (N, Ci, W, D)] 
         x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
         x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
         x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
         x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
         x5 = self.conv_and_pool(x,self.conv15) #(N,Co)
         x = torch.cat((x1, x2, x3, x4, x5), 1)
         x = self.dropout(x)  # (N, len(Ks)*Co)
         return x

    ''' Creates and trains a cnn neural network model. 
        X: a list of training data (string) 
        Y: a list of training labels (int)
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X, Y,batch_size=16,num_epochs=12,learning_rate=0.001):
#        start = time.time()
        # Parameters
        hidden_size = self.hidden_size
        dropout = self.dropout
#        print_every = 100
        #teacher_forcing_ratio = 0.9

        print("batch_size:", str(batch_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
#        print("encoding size:", str(encoding_size), "(0 means utterances are already encoded and input should be 3 dims)")

        #print("input_dim: ", str(input_dim))
#        print("output_dim: ", str(output_dim))

        # Set up optimizer and loss function
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
#        X = torch
#        X = X.to(cuda).contiguous()


        Y = to_categorical(Y)
        if use_cuda:
            self = self.to(cuda)
        Y = Y.astype('int') 
        X_len = X.shape[0]
#        num_labels = Y.shape[-1]
        steps = 0

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
#        if loss_function == 'cosine':
#            criterion = nn.CosineEmbeddingLoss()
#        elif loss_function == 'crossentropy':
#            criterion = nn.CrossEntropyLoss()
#        elif loss_function == 'mse':
#            criterion = nn.MSELoss()
#        elif loss_function == 'l1':
#            criterion = nn.L1Loss()
#        else:
#            print("WARNING: need to add loss function!")
        st = time.time()
        steps = 0
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = X[permutation]
            Yiter = Y[permutation]
    
            while True:
                 if i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]   
                 else:
                     batchX = Xiter[i:]
                     batchY = Yiter[i:] 
    #                 print('-------------%d----------------------'%i)
                 batchX = torch.tensor(batchX)
                 if use_cuda:
                     character_ids = batchX.to(cuda)
                 character_ids.contiguous()
    #                 print('type',type(character_ids))
    #                 print('size',character_ids.size())
                 Xtensor = self.elmo(character_ids)
                 Xtensor = Xtensor['elmo_representations'][0].float()
                 Ytensor = torch.from_numpy(batchY).long()
                 del batchX
                 del batchY
                 if use_cuda:
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
                 feature = Variable(Xtensor)
                 target = Variable(Ytensor)
                 del Xtensor
                 del Ytensor
                 i = i+batch_size
    
                 optimizer.zero_grad() 
                 output = self(feature)
    #             print(logit.size())    #
                 loss = F.cross_entropy(output, torch.max(target,1)[1])
                 loss.backward()
                 optimizer.step()
    
                 steps += 1
                 if i >= X_len:
                     break
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)

    def predict(self, testX, batch_size=16, keep_list=True, return_encodings=False,num_word=200):
        # Test the Model
        print_every = 1000
        pred = []
        i = 0
        length = len(testX)# .shape[0]
        if debug: print("testX len:", str(len(testX)))
        print("testing...")
        stime = time.time()
        testX = torch.tensor(testX)


        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        i = 0
    #    print(testX.size(0),'size')
        while True:
            if i+batch_size<testX.size(0):
    #        print(i)
                batchX = testX[i:i+batch_size]
            else: 
                batchX = testX[i:]
            if use_cuda:
                character_ids = batchX.to(cuda)
            character_ids.contiguous()
            Xtensor = self.elmo(character_ids)
            Xtensor = Xtensor['elmo_representations'][0].float()
            icd_var = self(Variable(Xtensor))
            
            icd_vec = logsoftmax(icd_var)
            for j in range(icd_vec.size(0)):
    #            print('icd_vec',icd_vec[i,:].size())
                icd_code = torch.max(icd_vec[j:j+1,:], 1)[1].data[0]
                icd_code = icd_code.item()
                y_pred.append(icd_code)
            i = i+batch_size
            if i >= testX.size(0):
                break
        print("testX shape: " + str(testX.shape))

        etime = time.time()
        print("testing took " + str(etime - stime) + " s")
        return y_pred
class CNN_Text(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=5, dropout=0.0, hidden_size = 100, USE_SERVER=False):
        super(CNN_Text, self).__init__()
#          if USE_SERVER:
#              wmodel,dim = load('/u/yanzhaod/data/va/mds+rct/narr+ice+medhelp.vectors.100')
#          else:
#              wmodel,dim = load('D:/projects/zhaodong/research/va/data/dataset/narr+ice+medhelp.vectors.100')
#          self.wmodel = wmodel
        D = embed_dim
        C = class_num
        Ci = 1
        Co = kernel_num
        Ks = kernel_sizes
        self.conv11 = nn.Conv2d(Ci, Co, (1, D))
        self.conv12 = nn.Conv2d(Ci, Co, (2, D))
        self.conv13 = nn.Conv2d(Ci, Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, Co, (5, D))
          #self.conv16 = nn.Conv2d(Ci, Co, (6, D))
          #self.conv17 = nn.Conv2d(Ci, Co, (7, D))
          #self.conv18 = nn.Conv2d(Ci, Co, (8, D))

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(Co*Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble 
	  
    def conv_and_pool(self, x, conv):
         x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
         x = F.max_pool1d(x, x.size(2)).squeeze(2)
         return x
     

    def forward(self, x):
        x = x.unsqueeze(1)  # (N, Ci, W, D)] 
        x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x5 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return x
    def fit(self, X, Y,batch_size=16,num_epochs=12,learning_rate=0.001):
        st = time.time()
        Y = to_categorical(Y)
        if debug: 
            print("X numpy shape: ", str(X.shape), "Y numpy shape:", str(Y.shape))
        X_len = X.shape[0]
        dim = X.shape[-1]
        num_labels = Y.shape[-1]
        num_epochs = num_epochs
        steps = 0
#          best_acc = 0
#          last_step = 0
#          log_interval = 1000
#          num_batches = math.ceil(X_len/batch_size)
        if use_cuda:
            self.to(cuda)
        
    # Train
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

        steps = 0
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = X[permutation]
            Yiter = Y[permutation]

            while i+batch_size < X_len:
                batchX = Xiter[i:i+batch_size]
                batchY = Yiter[i:i+batch_size]
                Xtensor = torch.from_numpy(batchX).float()
                Ytensor = torch.from_numpy(batchY).long()
                if use_cuda:
                    Xtensor = Xtensor.cuda()
                    Ytensor = Ytensor.cuda()
                feature = Variable(Xtensor)
                target = Variable(Ytensor)
                i = i+batch_size
                optimizer.zero_grad() 
                logit = self(feature)
    
                loss = F.cross_entropy(logit, torch.max(target,1)[1])
                loss.backward()
                optimizer.step()
    
            steps += 1
            print(timeSince(st))
    def predict(self, testX, batch_size=16, keep_list=True, return_encodings=False,num_word=200):
        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        i = 0
        while True: 
            Xtensor = torch.from_numpy(testX).float()
            if i+batch_size<Xtensor.size(0):
#        print(i)
                Xtensor = Xtensor[i:i+batch_size]
            else: 
                Xtensor = Xtensor[i:]
#            print(type(Xtensor))
            if use_cuda:
                Xtensor = Xtensor.to(cuda)
            Xtensor.contiguous()

            icd_var = self(Variable(Xtensor))   
            icd_vec = logsoftmax(icd_var)
            for j in range(icd_vec.size(0)):
#            print('icd_vec',icd_vec[i,:].size())
                icd_code = torch.max(icd_vec[j:j+1,:], 1)[1].data[0]
                icd_code = icd_code.item()
                y_pred.append(icd_code)
            i = i+batch_size
            if i >= testX.shape[0]:
                break
            
        return y_pred
         
     