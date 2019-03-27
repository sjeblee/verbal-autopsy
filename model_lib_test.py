
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

import pickle

numpy.set_printoptions(threshold=numpy.inf)
use_cuda = torch.cuda.is_available()
if use_cuda:
    torch.cuda.set_device(0)

use_cuda = True


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

# Concatenation of CNN and GRU model
class CNN_GRU_Text(nn.Module):
    def __init__(self, emb_dim_word,  emb_dim_char, class_num, hidden,max_num_char, kernel_num=200, kernel_sizes=5, dropout=0.0, ensemble=False, hidden_size=100):
        super(CNN_GRU_Text, self).__init__()
        
        #CNN
        D = emb_dim_word
        C = class_num
        Ci = 1
        self.C = class_num
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.ensemble = ensemble
        self.conv11 = nn.Conv2d(Ci, self.Co, (1, D))
        self.conv12 = nn.Conv2d(Ci, self.Co, (2, D))
        self.conv13 = nn.Conv2d(Ci, self.Co, (3, D))
        self.conv14 = nn.Conv2d(Ci, self.Co, (4, D))
        self.conv15 = nn.Conv2d(Ci, self.Co, (5, D))

        self.dropout = nn.Dropout(dropout)
        
        #GRU
        self.gru = nn.GRU(emb_dim_char,hidden_size)
        
        self.linear = nn.Linear(hidden_size,10)
        self.softmax = nn.LogSoftmax(dim=1)
        self.fc1 = nn.Linear(self.Co*self.Ks+max_num_char*10, C)

    def conv_and_pool(self, x, conv,n):
        x = F.relu(conv(x)).squeeze(3)  
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,a,b,hidden):
#        
        #Word
        a = a.squeeze(2).unsqueeze(1)   #[128, 1, 200, 100]=(N, Ci, Co, D)]
        a1 = self.conv_and_pool(a, self.conv11,1) # (N,Co)
        a2 = self.conv_and_pool(a, self.conv12,2) # (N,Co)
        a3 = self.conv_and_pool(a, self.conv13,3) # (N,Co)
        a4 = self.conv_and_pool(a, self.conv14,4) # (N,Co)
        a5 = self.conv_and_pool(a, self.conv15,5) # (N,Co)
        a = torch.cat((a1, a2, a3, a4, a5), 1)
        a = self.dropout(a)  # (N, len(Ks)*Co)
        #Char
        b,hidden = self.gru(b,hidden)
        b = self.linear(b)
        b = b.view(-1,b.size(1)*b.size(2))
        b = F.relu(b)
        input = torch.cat((a,b),1)      #1 is the horizontal concat
        output = self.fc1(input)
        output = self.softmax(output)
        
        #print(a.size(),b.size(),62)   #(1,1000), (1,1000)
        input = torch.cat((a,b),1)      #1 is the horizontal concat
        output = self.fc1(input)
        output = self.softmax(output)
        return output, hidden
        
class CNN_Text(nn.Module):

     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, ensemble=False, hidden_size = 100):
          super(CNN_Text, self).__init__()
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
#          print(x.size(),22222) #torch.Size([16, 1, 1000, 37])
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
class CNN_Text_Train(nn.Module):

     def __init__(self, input_dim, class_num, emb_dim, kernel_num=1000, kernel_sizes=6, dropout=0.0, ensemble=False, hidden_size = 100):
          super(CNN_Text_Train, self).__init__()
          D = emb_dim
          C = class_num
          Ci = 1
          Co = kernel_num
          Ks = kernel_sizes
          self.encoder = nn.Embedding(input_dim,emb_dim)
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
                                 # torch.Size([16, 1, 1000, 30])
#          print(conv(x).size(),22222)   #torch.Size([16, 1000, 1000, 30])
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
          
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
          return x
	  

     def forward(self, x):
#          print(x.size(),22222222) 
          x = self.encoder(x.long())
          x = x.squeeze(2).unsqueeze(1)
#          print(x.size(),11111111)
#          x = x.unsqueeze(1)  # (N, Ci, W, D)] 
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
class CNN_Comb_Text(nn.Module):

     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.25, ensemble=False, hidden_size = 100):
          super(CNN_Comb_Text, self).__init__()
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
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
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

'''
cnn_model with char input
'''
def char_cnn_model(X, Y, act=None, windows=[1,2,3,4,5], X2=[], learning_rate=0.001,batch_size=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.25, kernel_sizes=5):
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    Yarray = Y.astype('int') 
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    dim = Xarray.shape[-1]
    num_labels = Yarray.shape[-1]
    num_epochs = num_epochs
    steps = 0
    best_acc = 0
    last_step = 0
    log_interval = 1000
    num_batches = math.ceil(X_len/batch_size)
    
    model = CNN_Text(dim, num_labels,dropout=dropout, kernel_sizes=kernel_sizes)

    model = model.cuda()

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    model.train()
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
             logit = model(feature)
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
    return model
def char_cnn_model_train2(X, Y, emb_size=30, act=None, windows=[1,2,3,4,5], X2=[], learning_rate=0.001,batch_size=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.25, kernel_sizes=5):
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    Yarray = Y.astype('int') 
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    dim = Xarray.shape[-1]
    num_labels = Yarray.shape[-1]
    num_epochs = num_epochs
    steps = 0
    best_acc = 0
    last_step = 0
    log_interval = 1000
    num_batches = math.ceil(X_len/batch_size)
    model = CNN_Text_Train(dim, num_labels,emb_size,dropout=dropout, kernel_sizes=kernel_sizes)

    model = model.cuda()

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    model.train()
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
             logit = model(feature)
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
    return model
def char_cnn_model_train(X, Y, emb_size=30, act=None, windows=[1,2,3,4,5], X2=[], learning_rate=0.001,batch_size=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.25, kernel_sizes=5):
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    Yarray = Y.astype('int') 
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    dim = Xarray.shape[-1]
    num_labels = Yarray.shape[-1]
    num_epochs = num_epochs
    steps = 0
    best_acc = 0
    last_step = 0
    log_interval = 1000
    num_batches = math.ceil(X_len/batch_size)
    model = CNN_Text(dim, num_labels,emb_size,dropout=dropout, kernel_sizes=kernel_sizes)

    model = model.cuda()

    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
    model.train()
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
             logit = model(feature)
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
    return model

def cnn_comb_model(X,Y,emb_dim,batch_size=100,learning_rate=0.001,emb_dim_char=30,n_hidden=100,emb_dim_word=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
        # Train the CNN, return the model
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    Yarray = Y.astype('int') 
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    num_epochs = num_epochs
    num_labels = Yarray.shape[-1]
    steps = 0
    model = CNN_Comb_Text(emb_dim, num_labels,dropout=dropout, kernel_sizes=kernel_sizes)
    model.cuda()
#    print(Xarray.shape,X2array.shape,Yarray.shape,11111)  #(1580, 200, 100) (1580, 1484, 30) (1580, 9)
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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
             logit = model(Variable(Xtensor))
#             print(logit.size())  #[100,9]
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
    return model

def cnn_gru_model(X,X2,Y,all_categories,max_num_char,batch_size=100,learning_rate=0.001,emb_dim_char=30,n_hidden=100,emb_dim_word=100,num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
        # Train the CNN, return the model
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    X2array = numpy.asarray(X2).astype('float')
    Yarray = Y.astype('int') 
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    dim = Xarray.shape[-1]
    num_labels = Yarray.shape[-1]
    num_epochs = num_epochs
    steps = 0
    model = CNN_GRU_Text(emb_dim_word, emb_dim_char, len(all_categories), n_hidden, max_num_char)
    model.cuda()
#    print(Xarray.shape,X2array.shape,Yarray.shape,11111)  #(1580, 200, 100) (1580, 1484, 30) (1580, 9)
    # Train
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    steps = 0
   # cnn.train()
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        numpy.random.seed(seed=1)
        permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
        Xiter = Xarray[permutation]
        X2iter = X2array[permutation]
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
             Ytensor = Ytensor.cuda()
              
             feature = Variable(Xtensor)
             target = Variable(Ytensor)
             i = i+batch_size

             optimizer.zero_grad() 
             logit,hidden = model(Variable(Xtensor),Variable(X2tensor),None)
#             print(logit.size())  #[100,9]
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
    return model

''' Create and train a CNN model
    Hybrid features supported - pass structured feats as X2
    Does NOT support joint training yet
    returns: the CNN model
'''
def cnn_model(X, Y, act=None, windows=[1,2,3,4,5], X2=[], num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
    # Train the CNN, return the model
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    Yarray = Y.astype('int') 
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    dim = Xarray.shape[-1]
    num_labels = Yarray.shape[-1]
    num_epochs = num_epochs
    steps = 0
    best_acc = 0
    last_step = 0
    log_interval = 1000
    batch_size = 100
    num_batches = math.ceil(X_len/batch_size)
    learning_rate = 0.001
    cnn = CNN_Text(dim, num_labels,dropout=dropout, kernel_sizes=kernel_sizes)
    if use_cuda:
    	cnn = cnn.cuda()

    # Train
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    steps = 0
    cnn.train()
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
             logit = cnn(feature)
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
    return cnn

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

