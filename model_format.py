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

#class CNN_CONC(nn.Module):
#     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=5, dropout=0.0, ensemble=False, hidden_size=100,USE_SERVER=False):
#          super(CNN_CONC, self).__init__()
#          D = embed_dim
#          C = class_num
#          Ci = 1
#          Co = kernel_num
#          Ks = kernel_sizes
#          self.ensemble = ensemble
#          self.conv11 = nn.Conv2d(Ci, Co, (1, D))
#          self.conv12 = nn.Conv2d(Ci, Co, (2, D))
#          self.conv13 = nn.Conv2d(Ci, Co, (3, D))
#          self.conv14 = nn.Conv2d(Ci, Co, (4, D))
#          self.conv15 = nn.Conv2d(Ci, Co, (5, D))
#          #self.conv16 = nn.Conv2d(Ci, Co, (6, D))
#          #self.conv17 = nn.Conv2d(Ci, Co, (7, D))
#          #self.conv18 = nn.Conv2d(Ci, Co, (8, D))
#
#          self.dropout = nn.Dropout(dropout)
#          self.kernel_sizes = kernel_sizes
#          self.fc1 = nn.Linear(Co*Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble 
#	  
#     def conv_and_pool(self, x, conv):
#          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
#          x = F.max_pool1d(x, x.size(2)).squeeze(2)
#          return x
#	  
#
#     def forward(self, x):
#          x = x.unsqueeze(1)  # (N, Ci, W, D)] 
#          x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
#          x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
#          x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
#          x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
#          x5 = self.conv_and_pool(x,self.conv15) #(N,Co)
#          #x6 = self.conv_and_pool(x,self.conv16) #(N,Co)
#          #x7 = self.conv_and_pool(x,self.conv17) 
#          #x8 = self.conv_and_pool(x,self.conv18)
#          x = torch.cat((x1, x2, x3, x4, x5), 1)
#          
#          x = self.dropout(x)  # (N, len(Ks)*Co)
#          return x
#     def fit(self, X, Y,batch_size=16,num_epochs=12,learning_rate=0.001,DEBUG=False):
#         st=time.time()
#         Xarray = X
#         Yarray = Y.astype('int') 
#         print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))
#         X_len = Xarray.shape[0]
#         num_epochs = num_epochs
#         steps = 0
#         batch_size = 100
##         num_batches = math.ceil(X_len/batch_size)
#         learning_rate = 0.001
#
#         if use_cuda:
#         	self = self.cuda()
#    
#        # Train
#         optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
#         steps = 0
#         self.train()
#         for epoch in range(num_epochs):
#             print("epoch", str(epoch))
#             i = 0
#             numpy.random.seed(seed=1)
#             permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
#             Xiter = Xarray[permutation]
#             Yiter = Yarray[permutation]
#     
#             while i+batch_size < X_len:
#                  batchX = Xiter[i:i+batch_size]
#                  batchY = Yiter[i:i+batch_size]
#                  Xtensor = torch.from_numpy(batchX).float()
#                  Ytensor = torch.from_numpy(batchY).long()
#                  if use_cuda:
#                       Xtensor = Xtensor.cuda()
#                       Ytensor = Ytensor.cuda()
#                  feature = Variable(Xtensor)
#                  target = Variable(Ytensor)
#                  i = i+batch_size
#     
#                  optimizer.zero_grad() 
#                  logit = self(feature)
#     
#                  loss = F.cross_entropy(logit, torch.max(target,1)[1])
#                  loss.backward()
#                  optimizer.step()
#    
#                  steps += 1
#     
#            # Print epoch time
#             ct = time.time() - st
#             unit = "s"
#             if ct > 60:
#                 ct = ct/60
#                 unit = "m"
#             print("time so far: ", str(ct), unit)
#
#     def predict(self, testX, batch_size=16, keep_list=True, return_encodings=False,num_word=200):
##        testX = numpy.asarray(testX).astype('float')
#        y_pred = []
#        logsoftmax = nn.LogSoftmax(dim=1)
#        for x in range(len(testX)):
#           input_row = testX[x]
#           icd = None
#           if icd is None:
#               input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
#               input_tensor = input_tensor.contiguous().cuda()
#               icd_var = self.forward(Variable(input_tensor))
#               # Softmax and log softmax values
#               icd_vec = logsoftmax(icd_var)
#    #           icd_vec_softmax = softmax(icd_var)
#               icd_code = torch.max(icd_vec, 1)[1].data[0]
#           icd_code = icd_code.item()
#           y_pred.append(icd_code)
#        return y_pred  # Comment this line out if threshold is not in used. 
        
#import math
#import numpy
#import os
#import random
#import time
#import torch
#import torch.nn as nn
#from torch.autograd import Variable
#from torch import optim
#import torch.nn.functional as F
##import matplotlib.pyplot as plt
##import matplotlib.ticker as ticker
#
#import pickle
#
#numpy.set_printoptions(threshold=numpy.inf)
#use_cuda = torch.cuda.is_available()
#class CNN_CONC(nn.Module):
#     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=5, dropout=0.0, ensemble=False, hidden_size=100,USE_SERVER=False):
#          super(CNN_CONC, self).__init__()
#          D = embed_dim
#          C = class_num
#          Ci = 1
#          Co = kernel_num
#          Ks = kernel_sizes
#          self.ensemble = ensemble
#          self.conv11 = nn.Conv2d(Ci, Co, (1, D))
#          self.conv12 = nn.Conv2d(Ci, Co, (2, D))
#          self.conv13 = nn.Conv2d(Ci, Co, (3, D))
#          self.conv14 = nn.Conv2d(Ci, Co, (4, D))
#          self.conv15 = nn.Conv2d(Ci, Co, (5, D))
#
#          self.dropout = nn.Dropout(dropout)
#          self.kernel_sizes = kernel_sizes
#          self.fc1 = nn.Linear(Co*Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble 
#	  
#     def conv_and_pool(self, x, conv):
#          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
#          x = F.max_pool1d(x, x.size(2)).squeeze(2)
#          return x
#      
#     def forward(self, x):
#          x = x.unsqueeze(1)  # (N, Ci, W, D)] 
#          x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
#          x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
#          x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
#          x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
#          x5 = self.conv_and_pool(x,self.conv15) #(N,Co)
#          x = torch.cat((x1, x2, x3, x4, x5), 1)
#          x = self.dropout(x)  # (N, len(Ks)*Co)
#          return x
#      
#     def fit(self, X, Y,batch_size=16,num_epochs=12,learning_rate=0.001,DEBUG=False):
#        st = time.time()
#        Xarray = numpy.asarray(X).astype('float')
#        Yarray = Y.astype('int') 
#        print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))
#    
#        # Params
#        X_len = Xarray.shape[0]
#        num_epochs = num_epochs
#        num_labels = Yarray.shape[-1]
#        steps = 0
#        self.cuda()
#    #    print(Xarray.shape,X2array.shape,Yarray.shape,11111)  #(1580, 200, 100) (1580, 1484, 30) (1580, 9)
#        # Train
#        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
#    
#        steps = 0
#       # cnn.train()
#        for epoch in range(num_epochs):
#            print("epoch", str(epoch))
#            i = 0
#            numpy.random.seed(seed=1)
#            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
#            Xiter = Xarray[permutation]
#            Yiter = Yarray[permutation]
#    
#            while i+batch_size < X_len:
#                 batchX = Xiter[i:i+batch_size]
#                 batchY = Yiter[i:i+batch_size]
#                 Xtensor = torch.from_numpy(batchX).float()
#                 Xtensor.contiguous()
#    
#                 Ytensor = torch.from_numpy(batchY).long()
#    #             print("Ytensorsize",Ytensor.size())
#                 Xtensor = Xtensor.cuda()
#                 Ytensor = Ytensor.cuda()
#    
#                 target = Variable(Ytensor)
#                 i = i+batch_size
#    
#                 optimizer.zero_grad() 
#                 logit = self(Variable(Xtensor))
#    #             print(logit.size())  #[100,9]
#                 loss = F.cross_entropy(logit, torch.max(target,1)[1])
#                 print(loss)
#                 loss.backward()
#                 optimizer.step()
#    
#                 steps += 1
#     
#            # Print epoch time
#            ct = time.time() - st
#            unit = "s"
#            if ct > 60:
#                ct = ct/60
#                unit = "m"
#            print("time so far: ", str(ct), unit)
#     def predict(self, testX, batch_size=16, keep_list=True, return_encodings=False,num_word=200):
##        testX = numpy.asarray(testX).astype('float')
#        y_pred = []
#        logsoftmax = nn.LogSoftmax(dim=1)
#        for x in range(len(testX)):
#           input_row = testX[x]
#           icd = None
#           if icd is None:
#               input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
#               input_tensor = input_tensor.contiguous().cuda()
#               icd_var = self.forward(Variable(input_tensor))
#               # Softmax and log softmax values
#               icd_vec = logsoftmax(icd_var)
#    #           icd_vec_softmax = softmax(icd_var)
#               icd_code = torch.max(icd_vec, 1)[1].data[0]
#           icd_code = icd_code.item()
#           y_pred.append(icd_code)
#        return y_pred  # Comment this line out if threshold is not in used. 
class CNN_TEXT(nn.Module):

     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.25, ensemble=False, hidden_size = 100):
          super(CNN_TEXT, self).__init__()
          D = embed_dim
          C = class_num
          Ci = 1
          Co = kernel_num
          Ks = kernel_sizes
          self.ensemble = ensemble
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
#          print(x.size(),555555)
#          print(conv(x).size(),22222) #torch.Size([16, 1, 1000, 37])
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
#          print(x.size(),333333) #torch.Size([16, 1, 1000, 37])
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
#          print(x.size(),11111) #torch.Size([16, 1, 1000, 37])
          return x
	  

     def forward(self, x):
#          print(a.size(),b.size(),51) #torch.Size([1, 200, 100]) torch.Size([1, 1484, 30])
          x = x.unsqueeze(1)  # (N, Ci, W, D)] 
          x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
          x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
          x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
          x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
          x5 = self.conv_and_pool(x,self.conv15) #(N,Co)
          #x6 = self.conv_and_pool(x,self.conv16) #(N,Co)
          #x7 = self.conv_and_pool(x,self.conv17) 
          #x8 = self.conv_and_pool(x,self.conv18)
          x = torch.cat((x1, x2, x3, x4, x5), 1)
          
          x = self.dropout(x)  # (N, len(Ks)*Co)
          if self.ensemble == False: # Train CNN with no ensemble  
              logit = self.fc1(x)  # (N, C)
          else: # Train CNN with ensemble. Output of CNN will be input of another model
              logit = x
          return logit
     def fit(self,X,Y,emb_dim,batch_size=100,learning_rate=0.001,n_hidden=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
            st = time.time()
            Xarray = numpy.asarray(X).astype('float')
            Yarray = Y.astype('int') 
            print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))
        
            # Params
            X_len = Xarray.shape[0]
            num_epochs = num_epochs
            num_labels = Yarray.shape[-1]
            steps = 0
#            model = CNN_Comb_Text(emb_dim, num_labels,dropout=dropout, kernel_sizes=kernel_sizes)
            self.cuda()
        #    print(Xarray.shape,X2array.shape,Yarray.shape,11111)  #(1580, 200, 100) (1580, 1484, 30) (1580, 9)
            # Train
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
            steps = 0
           # cnn.train()
            for epoch in range(num_epochs):
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
                     Xtensor.contiguous()
        
                     Ytensor = torch.from_numpy(batchY).long()
        #             print("Ytensorsize",Ytensor.size())
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
        
                     target = Variable(Ytensor)
                     i = i+batch_size
        
                     optimizer.zero_grad() 
                     logit = self(Variable(Xtensor))
        #             print(logit.size())  #[100,9]
                     loss = F.cross_entropy(logit, torch.max(target,1)[1])
                     print(loss)
                     loss.backward()
                     optimizer.step()
        
                     steps += 1
         
                # Print epoch time
                ct = time.time() - st
                unit = "s"
                if ct > 60:
                    ct = ct/60
                    unit = "m"
                print("time so far: ", str(ct), unit)
     def predict(self,testX):
        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        for x in range(len(testX)):
           input_row = testX[x]
           icd = None
           if icd is None:
               input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
               input_tensor = input_tensor.contiguous().cuda()
#               self.cuda()
               icd_var = self(Variable(input_tensor))
               # Softmax and log softmax values
               icd_vec = logsoftmax(icd_var)
    #           icd_vec_softmax = softmax(icd_var)
               icd_code = torch.max(icd_vec, 1)[1].data[0]
           icd_code = icd_code.item()
           y_pred.append(icd_code)
        return y_pred  # Comment this line out if threshold is not in used. 