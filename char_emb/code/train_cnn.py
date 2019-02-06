
# coding: utf-8

# In[6]:

# Convert the csv of the translated narratives to an xml tree

from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import torch
import sys
from lxml import etree
import argparse
import calendar
import csv
import re
import subprocess
import time
import math 
from sklearn import metrics
import numpy as np
from random import shuffle
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report

# In[7]:
cuda = torch.device("cuda:1")
input_train = '/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml'
#input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat_spell.xml'
input_test = '/u/yanzhaod/data/va/mds+rct/test_adult_cat.xml'
out_file = "out_cnn_test.txt"           #output report text filename
out_model_filename = './model_cnn_adult.pt'   #output filename for the model

#parameters
n_conv_filters = 256
n_fc_neurons=1024
learning_rate = 0.0001
epochs = 20


def get_data(input):
    data={}                #key: category index, value:list of narratives
    seq_length = 0
    all_categories = []  
    tree = etree.parse(input)
    for e in tree.iter("cghr_cat"):     #iterate each "node" with tag=cghr_cat 
            if e.text not in data:      #e.text: the cghr_cat value
                data[e.text]=[]
                all_categories.append(e.text)
                
    for e in tree.iter("narrative","cghr_cat"):
        if e.tag == "narrative":        #get the content of narratives  
            value= e.text
            #preprocessing step for texts
            value = value.lower()
            value = re.sub('[^a-z0-9\s]','',value)
            if(seq_length) < len(value):   #obtain the maximum sequence length through loop
                seq_length = len(value)
            if e.tag == 'cghr_cat':     
                data[e.text].append(value)
        if e.tag == 'cghr_cat':             #once the narrative is obtained, assign it to the corresponding cghr_cat
            try:            
                    data[e.text].append(value)
            except:
                    print('Warning: need to append a value but the value does not exist')
    return data,seq_length,all_categories
data,seq_length,all_categories = get_data(input_train)
n_categories= len(all_categories)

vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
n_letters = len(vocab)
def letterToIndex(letter):
    try:
            return list(vocab).index(letter)
    except ValueError:
            print("Warning: the letter " + letter + "is not in the list")
            return 0

def letterToTensor(letter):
    tensor = torch.zeros([1, n_letters],device=cuda)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1>, where the second demension contains the category index
def lineToTensor(narrative):
    tensor = torch.zeros([seq_length,n_letters], device=cuda)  #may change dtype
    for li, letter in enumerate(narrative):
        tensor[li] = letterToTensor(letter)
    for i in range (1,seq_length-len(narrative)):
        tensor[len(narrative)+i][0]=0
    return tensor

narr = data['1'][0]
print(lineToTensor('poaspduodho sdjofjsdpoj35o34h5ofjdslfjotj').size(),1111)
# In[9]:

import torch.nn as nn
import math
class CNN(nn.Module):
    def __init__(self, n_classes=n_categories, input_length=seq_length, input_dim=37,
                 n_conv_filters=256,
                 n_fc_neurons=1024):
        super(CNN, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv1d(input_dim, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv2 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=7, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        self.conv3 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv1d(n_conv_filters, n_conv_filters, kernel_size=3, padding=0), nn.ReLU(),
                                   nn.MaxPool1d(3))
        dimension = math.floor((input_length-6)/3)
        dimension = math.floor((dimension-6)/3)
        dimension -= 6
        dimension = math.floor((dimension-2)/3)
        dimension = int(dimension)
        dimension *= n_conv_filters
        #dimension = int((input_length - 96) / 27 * n_conv_filters)
        self.fc1 = nn.Sequential(nn.Linear(dimension, n_fc_neurons), nn.Dropout(0.5))
        self.fc2 = nn.Sequential(nn.Linear(n_fc_neurons, n_fc_neurons), nn.Dropout(0.5))
        self.fc3 = nn.Linear(n_fc_neurons, n_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        if n_conv_filters == 256 and n_fc_neurons == 1024:
            self._create_weights(mean=0.0, std=0.05)
        elif n_conv_filters == 1024 and n_fc_neurons == 2048:
            self._create_weights(mean=0.0, std=0.02)   #!!!!may required modifications

    def _create_weights(self, mean=0.0, std=0.05):
        for module in self.modules():
            if isinstance(module, nn.Conv1d) or isinstance(module, nn.Linear):
                module.weight.data.normal_(mean, std)

    def forward(self, input):
        #input = self.encoder(input.long())
        input = input.transpose(1, 2)
        output = self.conv1(input)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = output.view(output.size(0), -1)
        output = self.fc1(output)
        output = self.fc2(output)
        output = self.fc3(output)
        output = self.softmax(output)
        return output

cnn = CNN(n_classes=n_categories,input_length=seq_length,input_dim=n_letters,n_conv_filters=n_conv_filters,n_fc_neurons=n_fc_neurons)
#cnn = CNN(seq_length,n_letters, n_hidden, n_categories, emb_size)
cnn.to(cuda)


# In[11]:
# Get the predicted category from the output layer (just get the maximum value)
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
#     print(top_n)
#     print(top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

#print(categoryFromOutput(output))


import random

def getTensors(category,line):
    # index = all_categories.index(category)
    # tensor = torch.zeros([n_categories],dtype=torch.long,device=cuda)
    # tensor[index] = 1
    # tensor.to(cuda)
    # 
    category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long, device=cuda)
    line_tensor = lineToTensor(line)
    line_tensor = line_tensor.to(cuda)
    category_tensor = category_tensor.to(cuda)

    return category, line, category_tensor, line_tensor
    
criterion = nn.NLLLoss()
#optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate,weight_decay=learning_rate/10)
optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate)
# In[28]:


def save():            #save the model to the .pt file (can reload it later!!)
   #save_filename = "./model_cnn.pt"
   torch.save(cnn, out_model_filename)
   print('Saved as %s' % out_model_filename)
   
def writeToFile(line,filename):
   if os.path.exists(filename):
           append_write = 'a' # append if already exists
   else:
           append_write = 'w' # make a new file if not

   f = open(filename,append_write)
   f.write(line + '\n')
   f.close()
   
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)

def trainIters(epochs):
    print_every = 100
    plot_every = 500
    iter = 0
    current_loss = 0
    all_losses = []
    start = time.time()
    
    batch_size = 128
    tensor_size = 0
    for k,v in data.iteritems():
            tensor_size += len(v)
    training_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    l = []
    for k,v in data.iteritems():
        for i in range(len(v)):
            category, line, category_tensor, line_tensor = getTensors(k,v[i])
            l.append([category_tensor,line_tensor])
    training_set = l
    training_generator = DataLoader(training_set, **training_params)
    #print(l[0][1].size(),1123)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    num_iter_per_epoch = len(training_generator)
    for epoch in range(epochs):
        for iter, batch in enumerate(training_generator):
            iter += batch_size
            label, feature = batch
            #print(feature.size())
            #feature = feature.cuda()
            #label = label.cuda()
            optimizer.zero_grad()
            predictions = cnn.forward(feature)
            loss = criterion(predictions, label)
            loss.backward()
            optimizer.step()
            
            # Print iter number, loss, name and guess
            if iter % print_every < batch_size:
                #guess, guess_i = categoryFromOutput(output)
                #correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters/epochs*100, timeSince(start), loss, line, guess, correct))
    
            # Add current loss avg to list of losses
            if iter % plot_every < batch_size:
                all_losses.append(current_loss / plot_every)
                current_loss = 0

    save()
    return cnn,l


def testTrainSet(model,train_data):
    total = 0
    counter = 0
    cat_pred,cat_true = [],[]
    for k,v in data.iteritems():
        for i in range(len(v)):
            category, line, category_tensor, line_tensor = getTensors(k,v[i])
            line_tensor = line_tensor.unsqueeze(0)
            output = model.forward(line_tensor)
            guess, guess_i = categoryFromOutput(output)
            cat_pred.append(guess)
            cat_true.append(category)

    f1score = metrics.f1_score(np.asarray(cat_true),np.asarray(cat_pred),average="weighted")
    print("f1score: ")
    print(f1score)
    return


def test(model):
    Tdata={}
    tree = etree.parse(input_test)
    for e in tree.iter('cghr_cat'):
        if e.text not in Tdata:
            Tdata[e.text] = []
            
    for e in tree.iter("narrative","cghr_cat"):
        if e.tag == "narrative":
            value= e.text
            value = value.lower()
            value = re.sub('[^a-z0-9\s]','',value)
            if seq_length < len(value):
                print("Warning: the length of the narr is longer than the sequence_length")
                print(seq_length,len(value))
                value = value[:seq_length]
        
        if e.tag == 'cghr_cat':
            Tdata[e.text].append(value)
            


    cat_pred,cat_true = [],[]
    for k,v in Tdata.iteritems():
        for i in range(len(v)):
            category,line,category_tensor,line_tensor = getTensors(k,v[i])
            line_tensor = line_tensor.unsqueeze(0)
            output = model.forward(line_tensor)
            guess,guess_i = categoryFromOutput(output)
            cat_pred.append(guess)
            cat_true.append(category)

    # print(cat_true,'true')
    # print(cat_pred,'pred')
    # print(len(cat_true),len(cat_pred))
    f1score = metrics.f1_score(np.asarray(cat_true),np.asarray(cat_pred),average="weighted")
    print("f1score: ")
    print(f1score)
    print(len(set(cat_true)))
    print(len(set(cat_pred)))
    print(len(all_categories))
    print(classification_report(np.asarray(cat_true),np.asarray(cat_pred),target_names=all_categories))
    return     

# In[92]:

if __name__ == '__main__':
    
    #model,train_data = trainIters(epochs)
    model = torch.load(out_model_filename)
    #testTrainSet(model,train_data=train_data)
    test(model)
    


