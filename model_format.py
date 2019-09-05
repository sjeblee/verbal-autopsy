1#!/usr/bin/python3

#import math
import numpy
#import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import optim
from keras.utils.np_utils import to_categorical
from torch.autograd import Variable
from word2vec import load
from word2vec import get
from utils import timeSince
import statistics
numpy.set_printoptions(threshold=numpy.inf)
debug = True
tdevice = 'cpu'
use_cuda = torch.cuda.is_available()
if use_cuda:
    cuda = torch.device("cuda:0")
import parameters
if 'elmo' in parameters.embedding:
    from allennlp.modules.elmo import Elmo, batch_to_ids
# GRU with GRU encoder, input: (conversations (1), utterances, words, embedding_dim)
class CNN_ELMO(nn.Module):
    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, hidden_size = 100, USE_SERVER=False):
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
    def fit(self, X, Y,batch_size=16,num_epochs=12,learning_rate=0.001,dropout=0.0):
#        start = time.time()
        # Parameters
        
        hidden_size = self.hidden_size
        dropout = self.dropout

        print("batch_size:", str(batch_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        if use_cuda:
            self = self.to(cuda)
        Y = Y.astype('int') 
        X_len = X.shape[0]
        steps = 0

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
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
                     character_ids = batchX.to(cuda).long()
                 else:
                     character_ids = batchX
                 character_ids.contiguous()
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
#                 print(feature.size())
                 output = self(feature)
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
            emacs("time so far: ", str(ct), unit)

    def predict(self, testX,testX2,batch_size=16, keep_list=True, return_encodings=False,num_word=200):
        # Test the Model
        i = 0
        if debug: print("testX len:", str(len(testX)))
        print("testing...")
        stime = time.time()
        testX = torch.tensor(testX).long()


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

class CNN_TEXT(nn.Module):

     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.25, ensemble=False, hidden_size = 100, USE_SERVER=False):
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

          self.dropout = nn.Dropout(dropout)
          self.fc1 = nn.Linear(Co*Ks, C) # Use this layer when train with only CNN model, i.e. No ensemble 
	  
     def conv_and_pool(self, x, conv):
#          print(x.size(),555555)
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
#          print(x.size(),333333) #torch.Size([16, 200，196-200])
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
#                     print(logit.size())  #[100,9]
                     loss = F.cross_entropy(logit, torch.max(target,1)[1])
#                     print(loss)
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
     def predict(self,testX,testX2):
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
class CNN_TEXT2(nn.Module):

     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, ensemble=False, hidden_size = 100, USE_SERVER=False):
          super(CNN_TEXT2, self).__init__()
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
          return self.fc1(x)  # (N, C)
     def fit(self,X,Y,emb_dim,batch_size=100,learning_rate=0.001,n_hidden=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
            st = time.time()
            Xarray = numpy.asarray(X).astype('float')
            Yarray = Y.astype('int') 
            print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))
        
            # Params
            X_len = Xarray.shape[0]
            steps = 0
            self.cuda()
        
            # Train
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        
            steps = 0
            self.train()
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
                     if use_cuda:
                          Xtensor = Xtensor.cuda()
                          Ytensor = Ytensor.cuda()
                     feature = Variable(Xtensor)
                     target = Variable(Ytensor)
                     i = i+batch_size
        
                     optimizer.zero_grad() 
                     logit = self(feature)
        #             print(logit.size())    #
                     loss = F.cross_entropy(logit, torch.max(target,1)[1])
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
class CNN_STRUCT(nn.Module):
     def __init__(self, emb_dim,emb_dim_feat, class_num, kernel_num=200, kernel_sizes=6, dropout=0.25, ensemble=False, hidden_size = 100, USE_SERVER=False):
          super(CNN_STRUCT, self).__init__()
          D = emb_dim
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
          self.fc1 = nn.Linear(Co*Ks+emb_dim_feat, C) # Use this layer when train with only CNN model, i.e. No ensemble 
          self.linear = nn.Linear(emb_dim_feat,emb_dim_feat)
     def conv_and_pool(self, x, conv):
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
#          print(x.size(),333333) #torch.Size([16, 1, 1000, 37])
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
#          print(x.size(),11111) #torch.Size([16, 1, 1000, 37])
          return x
	  

     def forward(self, a, b):
#          print(a.size(),b.size(),51) #torch.Size([1, 200, 100]) torch.Size([1, 1484, 30])
          a = a.unsqueeze(1)  # (N, Ci, W, D)] 
          a1 = self.conv_and_pool(a,self.conv11) #(N,Co)
          a2 = self.conv_and_pool(a,self.conv12) #(N,Co)
          a3 = self.conv_and_pool(a,self.conv13) #(N,Co)
          a4 = self.conv_and_pool(a,self.conv14) #(N,Co)
          a5 = self.conv_and_pool(a,self.conv15) #(N,Co)
          a = torch.cat((a1, a2, a3, a4, a5), 1)
          
          a = self.dropout(a)  # (N, len(Ks)*Co)
          b = self.linear(b)
          b = F.relu(b)
          input = torch.cat((a,b),1) 
          
          logit = self.fc1(input)  # (N, C)
          
#          input = torch.cat((logit,b),1)
#          print('input.size:'+str(input.size()))
          return logit
     def fit(self,X,X2,Y,emb_dim,batch_size=100,learning_rate=0.001,n_hidden=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
            st = time.time()
            Yarray = Y.astype('int') 
            print("X numpy shape: ", str(X.shape), "Y numpy shape:", str(Yarray.shape))
#            print('embeded dimension: '+str(emb_dim))
            # Params
            X_len = X.shape[0]
            num_epochs = num_epochs
            num_labels = Yarray.shape[-1]
            steps = 0
#            model = CNN_Comb_Text(emb_dim, num_labels,dropout=dropout, kernel_sizes=kernel_sizes)
            self.cuda()
        #    print(Xarray.shape,X2array.shape,Yarray.shape,11111)  #(1580, 200, 100) (1580, 1484, 30) (1580, 9)
            # Train
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            print(Y.shape,2)
            steps = 0
           # cnn.train()
            for epoch in range(num_epochs):
                print("epoch", str(epoch))
                i = 0
                numpy.random.seed(seed=1)
                permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
                Xiter = X[permutation]
                X2iter = X2[permutation]
                Yiter = Yarray[permutation]
        
                while i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchX2 = X2iter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]
                     Xtensor = torch.from_numpy(batchX).float()
                     Xtensor.contiguous()
                     X2tensor = torch.from_numpy(batchX2).float()
                     X2tensor.contiguous()        
                     Ytensor = torch.from_numpy(batchY).long()
        #             print("Ytensorsize",Ytensor.size())
                     Xtensor = Xtensor.cuda()
                     X2tensor = X2tensor.cuda()
#                     print(Xtensor.size(),X2tensor.size())
                     Ytensor = Ytensor.cuda()
        
                     target = Variable(Ytensor)
                     i = i+batch_size
        
                     optimizer.zero_grad() 
                     logit = self(Variable(Xtensor),Variable(X2tensor))
        #             print(logit.size())  #[100,9]
#                     print(torch.max(target,1)[1])
                     loss = F.cross_entropy(logit, torch.max(target,1)[1])
#                     print(loss)
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
     def predict(self,testX,testX2):
        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        for x in range(len(testX)):
           input_row = testX[x]
           input_row2 = testX2[x]
           icd = None
           if icd is None:
               input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
               input_tensor = input_tensor.contiguous().cuda()
               input_tensor2 = torch.from_numpy(numpy.asarray([input_row2]).astype('float')).float()
               input_tensor2 = input_tensor2.contiguous().cuda()
#               self.cuda()
               icd_var = self(Variable(input_tensor),Variable(input_tensor2))
               # Softmax and log softmax values
               icd_vec = logsoftmax(icd_var)
    #           icd_vec_softmax = softmax(icd_var)
               icd_code = torch.max(icd_vec, 1)[1].data[0]
           icd_code = icd_code.item()
           y_pred.append(icd_code)
        return y_pred  # Comment this line out if threshold is not in used. 
    
    
class STRUCT_ELMO(nn.Module):
    def __init__(self, embed_dim, emb_dim_feat,class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, hidden_size = 100, USE_SERVER=False):
          super(STRUCT_ELMO, self).__init__()
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
          self.fc1 = nn.Linear(Co*Ks+emb_dim_feat, C) # Use this layer when train with only CNN model, i.e. No ensemble 
          self.linear = nn.Linear(emb_dim_feat,emb_dim_feat)
          if debug:
              print("embedding dimension: "+str(embed_dim))
              print("number of classes: "+str(class_num))
    def conv_and_pool(self, x, conv):
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
          return x
    ''' Input is a list of strings
    '''
    def forward(self, a, b):
#          print(a.size(),b.size(),51) #torch.Size([1, 200, 100]) torch.Size([1, 1484, 30])
          a = a.unsqueeze(1)  # (N, Ci, W, D)] 
          a1 = self.conv_and_pool(a,self.conv11) #(N,Co)
          a2 = self.conv_and_pool(a,self.conv12) #(N,Co)
          a3 = self.conv_and_pool(a,self.conv13) #(N,Co)
          a4 = self.conv_and_pool(a,self.conv14) #(N,Co)
          a5 = self.conv_and_pool(a,self.conv15) #(N,Co)
          a = torch.cat((a1, a2, a3, a4, a5), 1)
          
          a = self.dropout(a)  # (N, len(Ks)*Co)
          b = self.linear(b)
          b = F.relu(b)
          input = torch.cat((a,b),1) 
          
          logit = self.fc1(input)  # (N, C)
          
#          input = torch.cat((logit,b),1)
#          print('input.size:'+str(input.size()))
          return logit

    ''' Creates and trains a cnn neural network model. 
        X: a list of training data (string) 
        Y: a list of training labels (int)
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X,X2,Y,batch_size=16,num_epochs=12,learning_rate=0.001,dropout=0.0):
        
#        start = time.time()
        # Parameters
        dropout = self.dropout
        print("batch_size:", str(batch_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if use_cuda:
            self = self.to(cuda)
        Y = Y.astype('int') 
        X_len = X.shape[0]
#        num_labels = Y.shape[-1]
        steps = 0

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        st = time.time()
        steps = 0
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = X[permutation]
            X2iter = X2[permutation]
            Yiter = Y[permutation]
    
            while True:
                 if i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchX2 = X2iter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]   
                 else:
                     batchX = Xiter[i:]
                     batchX2 = X2iter[i:]
                     batchY = Yiter[i:] 
    #                 print('-------------%d----------------------'%i)
                 batchX = torch.tensor(batchX)
                 if use_cuda:
                     character_ids = batchX.to(cuda).long()
                 character_ids.contiguous()
#                 print('type',type(character_ids))
#                 print('size',character_ids.size())
                 Xtensor = self.elmo(character_ids)
                 Xtensor = Xtensor['elmo_representations'][0].float()
                 
                 X2tensor = torch.from_numpy(batchX2).float()
                 X2tensor.contiguous()
                 
                 
                 Ytensor = torch.from_numpy(batchY).long()
                 del batchX
                 del batchY
                 del batchX2
                 if use_cuda:
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
                     X2tensor = X2tensor.cuda()
                 target = Variable(Ytensor)

                 i = i+batch_size
    
                 optimizer.zero_grad() 
                 output = self(Variable(Xtensor),Variable(X2tensor))
    #             print(logit.size())    #
#                 print(torch.max(target,1)[1])
                 loss = F.cross_entropy(output, torch.max(target,1)[1])
                 loss.backward()
                 optimizer.step()
                 del Xtensor
                 del X2tensor
                 del Ytensor
                 steps += 1
                 if i >= X_len:
                     break
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)

    def predict(self, testX,testX2,batch_size=16, keep_list=True, return_encodings=False,num_word=200):
        # Test the Model
        i = 0
        if debug: print("testX len:", str(len(testX)))
        print("testing...")
        stime = time.time()
#        testX = torch.tensor(testX)
#        testX2 = torch.tensor(testX2)

        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        i = 0
    #    print(testX.size(0),'size')
        while True:
            if i+batch_size<testX.shape[0]:
#                print(i)
                batchX = testX[i:i+batch_size]
                batchX2 = testX2[i:i+batch_size]
                batchX = torch.tensor(batchX).long()
            else: 
                batchX = testX[i:]
                batchX2 = testX2[i:]
                batchX = torch.tensor(batchX).long()
            if use_cuda:
                character_ids = batchX.to(cuda)
            character_ids.contiguous()
            Xtensor = self.elmo(character_ids)
            Xtensor = Xtensor['elmo_representations'][0].float()
            X2tensor = torch.from_numpy(numpy.asarray([batchX2]).astype('float')).float()
#            print(Xtensor.size(),X2tensor.size())
            X2tensor=X2tensor.squeeze(0)
#            print(Xtensor.size(),X2tensor.size())
            X2tensor = X2tensor.contiguous().cuda()         
            
            icd_var = self(Variable(Xtensor),Variable(X2tensor))
            
            icd_vec = logsoftmax(icd_var)
            for j in range(icd_vec.size(0)):
    #            print('icd_vec',icd_vec[i,:].size())
                icd_code = torch.max(icd_vec[j:j+1,:], 1)[1].data[0]
                icd_code = icd_code.item()
                y_pred.append(icd_code)
            i = i+batch_size
            if i >= testX.shape[0]:
                break
        print("testX shape: " + str(testX.shape))

        etime = time.time()
        print("testing took " + str(etime - stime) + " s")
        return y_pred
    
    

class BERT_CNN(nn.Module):
    def __init__(self,embed_dim, class_num, bert, config, kernel_num=768, kernel_sizes=6, dropout=0.25):
          super(BERT_CNN, self).__init__()
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
          self.fc1 = nn.Linear(768*5, C) # Use this layer when train with only CNN model, i.e. No ensemble 
          self.bert = bert
          self.kernel_num = kernel_num
          self.config=config
          self.rnn = nn.LSTM(bidirectional=True, num_layers=2, input_size=768, hidden_size=768//2, batch_first=True)
          self.fc = nn.Linear(768, class_num)
          self.class_num=class_num
    def conv_and_pool(self, x, conv):
         x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
         x = F.max_pool1d(x, x.size(2)).squeeze(2)
         return x
    def to_bert(self,x):
        output = torch.zeros((x.size(0),x.size(1),self.config.hidden_size))
        inds = x[0]
        inds = inds.unsqueeze(0).long()
        for i in range(x.size(0)):
            inds = x[i].long()
            if inds[-1] == 0:
                for j in range(len(inds)):
                    if inds[len(inds)-j-1] != 0:
                        break
            else:
                j = 0
            raw_inds = inds[:len(inds)-j]
            raw_inds = raw_inds.unsqueeze(0).long()
            with torch.no_grad():
                encoded_layers, _ = self.bert(raw_inds)
                output[i,:raw_inds.size(1),:] = encoded_layers[-1]

        return output
        
    def forward(self, x):
        '''
        x: (N, W, D). longTensor

        '''
            #(N,num_words,bert_hidden_size)
        x = x.unsqueeze(1)  # (N, Ci, W, D)] 
#        print(x.size(),22232111111)
        x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
        x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
        x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
        x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
        x5 = self.conv_and_pool(x,self.conv15) #(N,Co)
        x = torch.cat((x1, x2, x3, x4, x5), 1)
        x = self.dropout(x)  # (N, len(Ks)*Co)
        return self.fc1(x)
    
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
           
           #test
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = Xarray[permutation]
            batchX = Xiter[0:1]
            Xtensor = torch.from_numpy(batchX)
            Xtensor = Xtensor.cuda()
#            print(self.to_bert(Xtensor).float(),22212212)
            print_every = 10
           
           
            for epoch in range(num_epochs):
                print("epoch", str(epoch))
                i = 0
                numpy.random.seed(seed=1)
                permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
                Xiter = Xarray[permutation]
                Yiter = Yarray[permutation]
                loss_total = []
                while i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]
                     Xtensor = torch.from_numpy(batchX)
                     Xtensor = Xtensor.cuda()
                     if i == 0:
                         print(Xtensor.size())
#                     print(Xtensor.size(),1111)
                     Xtensor = self.to_bert(Xtensor).float()
#                     print(Xtensor.size(),22222)
                     if i == 0:
                         print(Xtensor.size())
                     Xtensor.contiguous()
        
                     Ytensor = torch.from_numpy(batchY).long()
        #             print("Ytensorsize",Ytensor.size())
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
        
                     target = Variable(Ytensor)
                     i = i+batch_size
        
                     optimizer.zero_grad() 
                     
                     logit = self(Variable(Xtensor))
#                     print(logit.size(),22222)  #[batch_size,class_num]
#                     print(target.size(),23213)   #[16,9]
                     loss = F.cross_entropy(logit, torch.max(target,1)[1])
#                     loss_fct = torch.nn.BCEWithLogitsLoss()
#                     print(logit.view(-1, self.class_num).size(),torch.max(target,1)[1].size())
#                     loss = loss_fct(logit, torch.max(target,1)[1])
                     loss_total.append(float(loss))
                     if len(loss_total) >= print_every:
                         print('Loss: '+str(statistics.mean(loss_total)))
                         loss_total = []
#                     print(loss)
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
    def predict(self,testX,testX2):
        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        for x in range(len(testX)):
           input_row = testX[x]
           icd = None
           if icd is None:
               input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
               input_tensor = input_tensor.cuda()
               input_tensor = self.to_bert(input_tensor).float()
               input_tensor = input_tensor.contiguous().cuda()
#               self.cuda()
               icd_var = self(Variable(input_tensor))
               # Softmax and log softmax values
               icd_vec = logsoftmax(icd_var)
    #           icd_vec_softmax = softmax(icd_var)
               icd_code = torch.max(icd_vec, 1)[1].data[0]
           icd_code = icd_code.item()
           y_pred.append(icd_code)
#           print(y_pred)
        return y_pred  # Comment this line out if threshold is not in used. 
    
    
class STRUCT_BERT(nn.Module):
    def __init__(self, embed_dim, emb_dim_feat,class_num, bert, config, kernel_num=200, kernel_sizes=6, dropout=0.0, hidden_size = 100, USE_SERVER=False):
          super(STRUCT_BERT, self).__init__()
          options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
          if USE_SERVER == True:
              weight_file = '/u/yanzhaod/data/elmo_pubMed_only.hdf5'
          else:
              weight_file = 'D:/projects/zhaodong/research/elmo_pubMed_only.hdf5'
          self.hidden_size = hidden_size
#          self.elmo = Elmo(options_file, weight_file, 1, dropout=0).to(tdevice)
          
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
          self.fc1 = nn.Linear(embed_dim*Ks+emb_dim_feat, C) # Use this layer when train with only CNN model, i.e. No ensemble 
          self.linear = nn.Linear(emb_dim_feat,emb_dim_feat)
          self.config=config
          self.class_num=class_num
          self.bert=bert
          if debug:
              print("embedding dimension: "+str(embed_dim))
              print("number of classes: "+str(class_num))
    def conv_and_pool(self, x, conv):
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
          return x
    def to_bert(self,x):
        output = torch.zeros((x.size(0),x.size(1),self.config.hidden_size))
        inds = x[0]
        inds = inds.unsqueeze(0).long()
        for i in range(x.size(0)):
            inds = x[i].long()
            if inds[-1] == 0:
                for j in range(len(inds)):
                    if inds[len(inds)-j-1] != 0:
                        break
            else:
                j = 0
            raw_inds = inds[:len(inds)-j]
            raw_inds = raw_inds.unsqueeze(0).long()
            with torch.no_grad():
                encoded_layers, _ = self.bert(raw_inds)
                output[i,:raw_inds.size(1),:] = encoded_layers[-1]

        return output
    ''' Input is a list of strings
    '''
    def forward(self, a, b):
#          print(a.size(),b.size(),51) #torch.Size([1, 200, 100]) torch.Size([1, 1484, 30])
          a = a.unsqueeze(1)  # (N, Ci, W, D)] 
          a1 = self.conv_and_pool(a,self.conv11) #(N,Co)
          a2 = self.conv_and_pool(a,self.conv12) #(N,Co)
          a3 = self.conv_and_pool(a,self.conv13) #(N,Co)
          a4 = self.conv_and_pool(a,self.conv14) #(N,Co)
          a5 = self.conv_and_pool(a,self.conv15) #(N,Co)
          a = torch.cat((a1, a2, a3, a4, a5), 1)
          
          a = self.dropout(a)  # (N, len(Ks)*Co)
          b = self.linear(b)
          b = F.relu(b)
          input = torch.cat((a,b),1) 
          
          logit = self.fc1(input)  # (N, C)
          
#          input = torch.cat((logit,b),1)
#          print('input.size:'+str(input.size()))
          return logit

    ''' Creates and trains a cnn neural network model. 
        X: a list of training data (string) 
        Y: a list of training labels (int)
        WARNING: Currently you can't use the encoding layer and use_prev_labels at the same time
    '''
    def fit(self, X,X2,Y,batch_size=16,num_epochs=12,learning_rate=0.001,dropout=0.0):
#        start = time.time()
        # Parameters
        dropout = self.dropout
        print("batch_size:", str(batch_size), "dropout:", str(dropout), "epochs:", str(num_epochs))
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if use_cuda:
            self = self.to(cuda)
        Y = Y.astype('int') 
        X_len = X.shape[0]
#        num_labels = Y.shape[-1]
        steps = 0

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        st = time.time()
        steps = 0
        for epoch in range(num_epochs):
            print("epoch", str(epoch))
            i = 0
            numpy.random.seed(seed=1)
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = X[permutation]
            X2iter = X2[permutation]
            Yiter = Y[permutation]
    
            while True:
                 if i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchX2 = X2iter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]   
                 else:
                     batchX = Xiter[i:]
                     batchX2 = X2iter[i:]
                     batchY = Yiter[i:] 
    #                 print('-------------%d----------------------'%i)
#                 batchX = torch.tensor(batchX)
                 Xtensor = torch.from_numpy(batchX)
                 Xtensor = Xtensor.cuda()
                 Xtensor = self.to_bert(Xtensor).float()
                 Xtensor.contiguous()   
                 
                 X2tensor = torch.from_numpy(batchX2).float()
                 X2tensor.contiguous()
                 
                 
                 Ytensor = torch.from_numpy(batchY).long()
                 del batchX
                 del batchY
                 del batchX2
                 if use_cuda:
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
                     X2tensor = X2tensor.cuda()
                 target = Variable(Ytensor)

                 i = i+batch_size
    
                 optimizer.zero_grad() 
                 output = self(Variable(Xtensor),Variable(X2tensor))
    #             print(logit.size())    #
#                 print(torch.max(target,1)[1])
                 loss = F.cross_entropy(output, torch.max(target,1)[1])
                 loss.backward()
                 optimizer.step()
                 del Xtensor
                 del X2tensor
                 del Ytensor
                 steps += 1
                 if i >= X_len:
                     break
            ct = time.time() - st
            unit = "s"
            if ct > 60:
                ct = ct/60
                unit = "m"
            print("time so far: ", str(ct), unit)

    def predict(self, testX,testX2,batch_size=16, keep_list=True, return_encodings=False,num_word=200):
        # Test the Model
        i = 0
        if debug: print("testX len:", str(len(testX)))
        print("testing...")
        stime = time.time()
#        testX = torch.tensor(testX)
#        testX2 = torch.tensor(testX2)

        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        i = 0
    #    print(testX.size(0),'size')
        while True:
            if i+batch_size<testX.shape[0]:
#                print(i)
                batchX = testX[i:i+batch_size]
                batchX2 = testX2[i:i+batch_size]
#                batchX = torch.tensor(batchX).long()
            else: 
                batchX = testX[i:]
                batchX2 = testX2[i:]
#                batchX = torch.tensor(batchX).long()
            Xtensor = torch.from_numpy(batchX)
            Xtensor = Xtensor.cuda()
            Xtensor = self.to_bert(Xtensor).float()
            Xtensor.contiguous()      
            Xtensor = Xtensor.cuda()
            X2tensor = torch.from_numpy(numpy.asarray([batchX2]).astype('float')).float()
            X2tensor=X2tensor.squeeze(0)
            X2tensor = X2tensor.contiguous().cuda()         
            
            icd_var = self(Variable(Xtensor),Variable(X2tensor))
            
            icd_vec = logsoftmax(icd_var)
            for j in range(icd_vec.size(0)):
    #            print('icd_vec',icd_vec[i,:].size())
                icd_code = torch.max(icd_vec[j:j+1,:], 1)[1].data[0]
                icd_code = icd_code.item()
                y_pred.append(icd_code)
            i = i+batch_size
            if i >= testX.shape[0]:
                break
        print("testX shape: " + str(testX.shape))

        etime = time.time()
        print("testing took " + str(etime - stime) + " s")
        return y_pred
        
class BERT_LSTM(nn.Module):
    def __init__(self,embed_dim, class_num, bert, config, kernel_num=768, kernel_sizes=6, dropout=0.1,hidden_size=40,input_size=768):
          super(BERT_LSTM, self).__init__()
          D = embed_dim
          C = class_num
          Ci = 1
          Co = kernel_num
          Ks = kernel_sizes
          self.dropout = nn.Dropout(dropout)
          self.bert = bert
          self.kernel_num = kernel_num
          self.config=config
          self.lstm = nn.GRU(input_size,hidden_size, bidirectional=True,dropout=dropout)
          self.fc = nn.Linear(hidden_size*2, class_num)
          self.class_num=class_num
    def conv_and_pool(self, x, conv):
         x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
         x = F.max_pool1d(x, x.size(2)).squeeze(2)
         return x
    def to_bert(self,x):
        inds = x.squeeze(0).long()   #inds[200]
#        print(inds.size(),222)
        if inds[-1] == 0:
            for j in range(len(inds)):
                if inds[len(inds)-j-1] != 0:
                    break
        else:
            j = 0
        raw_inds = inds[:len(inds)-j]
        raw_inds = raw_inds.unsqueeze(0).long()  #[1,num_words,768]
        with torch.no_grad():
            encoded_layers, _ = self.bert(raw_inds)

        return encoded_layers[-1]
        
    def forward(self, x):
        '''
        x: (N, W, D). longTensor

        '''

#        x,_ = self.lstm(x.view(x.size(0),1,-1))
#        print(x.size(),12313)
        x,_ = self.lstm(x)
#        print(x.size(),12231323213)
#        print(self.fc(x).size(),12231323213)
#        print(self.fc(x)[:,-1,:].size(),12231323213)
        #return self.fc(x)[:,-1,:]
        return self.fc(x[-1])
    
    def fit(self,X,Y,emb_dim,batch_size=100,learning_rate=0.001,n_hidden=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.1, kernel_sizes=5):
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
           
           #test
            permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
            Xiter = Xarray[permutation]
            batchX = Xiter[0:1]
            Xtensor = torch.from_numpy(batchX)
            Xtensor = Xtensor.cuda()
#            print(self.to_bert(Xtensor).float(),22212212)
            print_every = 50
           
            batch_size = 1
            for epoch in range(num_epochs):
                print("epoch", str(epoch))
                i = 0
                numpy.random.seed(seed=1)
                permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
                Xiter = Xarray[permutation]
                Yiter = Yarray[permutation]
                loss_total = []
                while i+batch_size < X_len:
                     batchX = Xiter[i:i+batch_size]
                     batchY = Yiter[i:i+batch_size]
                     Xtensor = torch.from_numpy(batchX)
                     Xtensor = Xtensor.cuda()
                     #Xtensor: [1,200]
                     Xtensor = Xtensor.squeeze(0)  #[1,200]
                     #Xtensor: [200]
                     
                     if i == 0:
                         print(Xtensor.size())
                     Xtensor = self.to_bert(Xtensor).float() #1,numwords,768 
                     #Xtensor: [1,numwords,768]
                     
                     Xtensor = Xtensor.squeeze(0).unsqueeze(1)
                     #Xtensor: [numwords,1,768]

                     if i == 0:
                         print(Xtensor.size())
                     Xtensor.contiguous()
        
                     Ytensor = torch.from_numpy(batchY).long()
        #             print("Ytensorsize",Ytensor.size())
                     Xtensor = Xtensor.cuda()
                     Ytensor = Ytensor.cuda()
        
                     target = Variable(Ytensor)
                     i = i+batch_size                     
        
                     optimizer.zero_grad() 
#                     print(Xtensor.size())
                     logit = self(Variable(Xtensor))
#                     print(logit.size(),22222)  #[batch_size,class_num]
#                     print(torch.max(target,1)[1].size(),23213)   #[9]
                     loss = F.cross_entropy(logit, torch.max(target,1)[1])
#                     loss_fct = torch.nn.BCEWithLogitsLoss()
#                     print(logit.view(-1, self.class_num).size(),torch.max(target,1)[1].size())
#                     loss = loss_fct(logit, torch.max(target,1)[1])
                     loss_total.append(float(loss))
                     if len(loss_total) >= print_every:
                         print('Loss: '+str(statistics.mean(loss_total)))
                         loss_total = []
#                     print(loss)
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
    def predict(self,testX,testX2):
        y_pred = []
        logsoftmax = nn.LogSoftmax(dim=1)
        for x in range(len(testX)):
           input_row = testX[x]
#           print(input_row)
           icd = None
           if icd is None:
               input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
               input_tensor = input_tensor.cuda()
               input_tensor = self.to_bert(input_tensor).float()
               input_tensor = input_tensor.contiguous().cuda()
#               self.cuda()
#               print(input_tensor.size(),111)
               input_tensor = input_tensor.squeeze(0).unsqueeze(1)
               icd_var = self(Variable(input_tensor))
               # Softmax and log softmax values
               icd_vec = logsoftmax(icd_var)
#               print(icd_vec.size())
    #           icd_vec_softmax = softmax(icd_var)
               icd_code = torch.max(icd_vec, 1)[1].data[0]
           icd_code = icd_code.item()
           y_pred.append(icd_code)
#           print(y_pred)
        return y_pred  # Comment this line out if threshold is not in used. 
    