
# coding: utf-8

# In[1]:

from __future__ import unicode_literals, print_function, division

import torch
import csv
import torch.nn as nn
import re



from io import open
import glob
import os

import sys
from lxml import etree
import argparse
import calendar
from torch.utils.data import DataLoader
import torch.nn.functional as F
import subprocess
import time
import math 
from sklearn.metrics import f1_score
import numpy as np
from random import shuffle
from word2vec import load
from word2vec import get


###########################################################
#**********************Parameters************************
#cuda -> the selected gpu
#input_train -> (string) the path of the input training file
#out_model_filename -> (string) the stored net file that can be reloaded lator
#out_text_filename -> (string) the output text file showing some relevant info
#out_results_filename -> (string) the output file for buiding the confusion matrix 
#n_hidden -> (int) hidden size of the neural net, usually 64 or 128
#emb_dim_char -> (int) embedding size
#learning_rate -> (float) learning rate
#**********************Variables************************
#data  ->   (dictionary), key=(string) cghr category; value=(tuple) ((string) MG_ID, (string)narrative text) 
#vocab  ->  (string list), contains each of the letters
#n_letters -> (int), number of letters
###########################################################

# In[3]:

cuda = torch.device("cuda:0")
        

input_train = '/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml'
input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat_spell.xml'
#input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat.xml'
out_model_filename = "./char_emb/code/output/model_child_gru_128.pt"
out_text_filename = "char_emb/code/output/out_adult_test_128.txt"
out_results_filename = 'char_emb/code/output/out_adult_results.txt'

# Hidden size
n_hidden = 64            

# Embedding size
emb_dim_char =  30

# Learning rate
learning_rate = 0.0001

# Epochs
epochs = 30

choice = "both"
from utils import get_dic
from utils import get_data

data,all_categories = get_data(input_train)   #get the data
n_categories= len(all_categories)
vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
n_letters = len(vocab)
n_iters = 0
for k,v in data.iteritems():
    n_iters += len(v)
print("size of the narratives: %d" %n_iters)
print("vocab: %s" %vocab)
char_emb_fname = 'char_emb/code/char_emb_30.txt'
emb_dic = get_dic(char_emb_fname,vocab)

l = []
max_num_word = 0
max_num_char = 0
for k in data:
    v = data[k]
    for i in range(len(v)):
        if len(v[i][1]) > max_num_char:
            max_num_char = len(v[i][1])
        word_list = re.split(' ',v[i][1])
        l.append((k,(v[i][0],word_list)))
        if len(word_list) > max_num_word:
            max_num_word = len(word_list)
shuffle(l)
print("max_num_word: %s; max_num_char:%s" %(max_num_word,max_num_char))

def letterToTensor(letter):
    tensor = torch.tensor(emb_dic[letter],device=cuda)
    return tensor
def lineToTensor(narrative):
    tensor = torch.zeros([max_num_char,emb_dim_char],device=cuda)
    for li, letter in enumerate(narrative):
        tensor[li] = letterToTensor(letter)
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def getTensors(category,line):
    category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long,device=cuda)
    #category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    category_tensor = category_tensor.to(cuda)
    return category, line, category_tensor, line_tensor
def getWordTensors(word_list):
    tensor = torch.zeros([max_num_word, emb_dim_word],device=cuda)
    for i in range(max_num_word):
        if i < len(word_list):
            tensor[i] = torch.tensor(get(word_list[i],wmodel),dtype=torch.long)
    #tensor = torch.unsqueeze(tensor,1)   #d 
    return tensor

# from utils import lineToTensorChar
# from utils import getTensorsChar
#max_num_word = 200   
wmodel,dim = load('/u/yanzhaod/data/narr_ice_medhelp.vectors.100')
emb_dim_word = len(get('have',wmodel))
class_num = n_categories

narr = data[data.keys()[0]][1][1]
input = lineToTensor(narr)
input=input.to(cuda)
from utils import CNN_Text

# class GRU_Text(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, emb_size):
#         super(GRU_Text, self).__init__()
#         
#         self.hidden_size = hidden_size
#         
#         self.encoder = nn.Embedding(input_size,emb_size)
#         self.gru = nn.GRU(emb_size*input_size,hidden_size)
#         self.linear = nn.Linear(hidden_size,output_size)
#         self.softmax = nn.LogSoftmax(dim=1)
# 
#     def forward(self, input, hidden):
#         input = self.encoder(input.long())    #(batch,num,1,emb)
# 
#         input = input.transpose(0,2)
#         input = input.transpose(1,2) #(1,batch,num,emb)
# 
#         input = input.view(input.size(0),input.size(1),input.size(2)*input.size(3))#(1,batch,num*emb)
#   
#         b,hidden = self.gru(input,hidden)  #(1,batch,hidden)
# 
#         b = self.linear(b)#(1,batch,output_size)
#         #output = self.softmax(output)
#         b = b.transpose(0,1)  #(batch,1,hidden)
#         b = b.squeeze(1)     #(batch,hidden)
#         b = self.softmax(b)
#         return b, hidden
#         
#     def initHidden(self):
#         return torch.zeros([1, self.hidden_size],device=cuda)
class CNN_GRU_Text(nn.Module):
    def __init__(self, emb_dim_word,  emb_dim_char, class_num, hidden, kernel_num=200, kernel_sizes=5, dropout=0.0, ensemble=False, hidden_size=100):
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

    def forward(self,a,b, hidden):
        #print(a.size(),b.size(),51)   #(batch,max_num_word,100); (batch,max_num_char,30)
        #Word
        a = a.unsqueeze(1)  # (N, Ci, W, D)]
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
        return output, hidden
criterion = nn.NLLLoss()
if choice == "gru":
    gru = GRU_Text(max_num_char,n_hidden,n_categories,emb_dim_char)
    gru.to(cuda)
    optimizer_char = torch.optim.Adam(gru.parameters(), lr=learning_rate)

if choice == "cnn":
    cnn = CNN_Text(emb_dim_word,n_categories)
    cnn.to(cuda)
    optimizer_word = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
if choice == "both":
    model = CNN_GRU_Text(emb_dim_word, emb_dim_char, n_categories, n_hidden, kernel_num=max_num_word)
    model.to(cuda)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#print(getWordTensors('i have a dog'.split()).size(),1111)

def save(model):
    torch.save(model, out_model_filename)
    print('Saved as %s' % out_model_filename)
    
from utils import writeToFile
from utils import timeSince



def train_iter():
    plot_every = 5000
    batch_size = 128
    print_every = 10*batch_size
    current_loss = 0
    all_losses = []
    start = time.time()
    iter = 0
    print_every = 10
    if choice == "both":
        training_set = []

        for e in l:
            k,v = e[0],e[1]
            category, line, category_tensor, line_tensor_char = getTensors(k,' '.join(v[1]))
            line_tensor_char = line_tensor_char.to(cuda)
            line_tensor_word = getWordTensors(v[1])
            training_set.append([category_tensor,line_tensor_char,line_tensor_word])
        training_params = {"batch_size": batch_size,
                        "shuffle": True,
                        "num_workers": 0}
        training_generator = DataLoader(training_set, **training_params)
        count = 0
        print("begin training:.....")
        for epoch in range(epochs):
            count += 1
            for iter, batch in enumerate(training_generator):
                count += batch_size
                category_tensor, line_tensor_char, line_tensor_word = batch
                if line_tensor_char.size() == (0,):
                    continue 
                #train

                optimizer.zero_grad()  
                #print(line_tensor_word.size(),line_tensor_char.size())  #(batch,max_len_word,word_emb)(batch,max_len_char,char_emb)
    
                output,hidden = model.forward(line_tensor_word,line_tensor_char,None)
                #print(output.size(),category_tensor.size(),2)  ((batch,9);(batch,))
                loss = criterion(output, category_tensor)
                loss.backward()
                optimizer.step()
                if count % print_every == 0:
                    print('%d %d%% (%s)' % (count*batch_size, count*batch_size / n_iters/epochs*100, timeSince(start))) 
        save(model)
        return model
    elif choice == "gru":
        training_set_char = []
        for e in l:
            k,v = e[0],e[1]
            category, line, category_tensor, line_tensor_char = getTensorsChar(k,' '.join(v[1]),all_categories,max_num_char,vocab,cuda)
            line_tensor_char = line_tensor_char.to(cuda)
            #print(line_tensor_char.size())
            training_set_char.append([category_tensor,line_tensor_char])
        training_params = {"batch_size": batch_size,
                        "shuffle": True,
                        "num_workers": 0}
        training_generator = DataLoader(training_set_char, **training_params)
        count = 0
        for epoch in range(epochs):
            for iter, batch in enumerate(training_generator):
                count += 1
                category_tensor, line_tensor_char= batch
                if line_tensor_char.size() == (0,):
                    continue 
                #train

                optimizer_char.zero_grad() 
                #print(line_tensor_char.size())
                output,hidden = gru(line_tensor_char,None)
                loss = criterion(output, category_tensor)
                loss.backward()
                optimizer_char.step()
                if count % print_every == 0:
                    print('%d %d%% (%s)' % (count*batch_size, count*batch_size / n_iters/epochs*100, timeSince(start)))  
        save(gru)
        return gru
    else:
        
        training_set_word = []
        for e in l:
            category,v = e[0],e[1]
            category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long,device=cuda)
            category_tensor = category_tensor.to(cuda)
            line_tensor_word = getWordTensors(v[1])
            training_set_word.append([category_tensor,line_tensor_word])
        training_params = {"batch_size": batch_size,
                        "shuffle": True,
                        "num_workers": 0}
        training_generator = DataLoader(training_set_word, **training_params)
        
        count = 0
        for epoch in range(epochs):
            for iter, batch in enumerate(training_generator):
                count += 1
                category_tensor, line_tensor_word= batch
                #train
                optimizer_word.zero_grad() 
                output = cnn(line_tensor_word)
                loss = criterion(output, category_tensor)
                loss.backward()
                optimizer_word.step() 
                if iter % print_every == 0:
                        print('%d %d%% (%s)' % (count*batch_size, count*batch_size / n_iters/epochs*100, timeSince(start))) 
                    
        save(cnn)
        return cnn
def test_cnn(model):
    tdata,all_categories = get_data(input_test)
    result = []
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 1000
    start = time.time()
    for category,v in tdata.iteritems():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long,device=cuda)
            category_tensor = category_tensor.to(cuda)
            line_tensor_word = getWordTensors(v[1])
            
            MG_ID = v[i][0]
            
            line_tensor_word = line_tensor_word.unsqueeze(0)

            output = cnn(line_tensor_word)
            guess, guess_i = categoryFromOutput(output)
            result.append({'Correct_ICD':category,'Predicted_ICD':guess,'MG_ID':MG_ID})
            #print(category,guess)         #uncomment this line for detailed label/prediction pairs
            cat_pred.append(guess)
            cat_true.append(category)
    print('----------------------------------------------')
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    print(f1score)
    writeToFile("f1score: " + str(f1score),out_text_filename)
    for i in range(len(result)):
        result[i] = str(result[i])
    writeToFile('\n'.join(result),out_results_filename)
    return
def test_gru(model):
    tdata,all_categories = get_data(input_test)
    result = []
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 1000
    start = time.time()
    for k,v in tdata.iteritems():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            try:
                category, line, category_tensor, line_tensor_char = getTensorsChar(k,v[i][1],all_categories,max_num_char,vocab,cuda)
            except ValueError:
                print('----------------outsided text----------------')
                print(v[i][1])
                print('\t' in v[i][1])
                iter -= 1
                continue
            if line_tensor_char.size() == (0,):
                continue 
            
            MG_ID = v[i][0]
            
            # line_tensor_word = line_tensor_word.unsqueeze(0)
            line_tensor_char = line_tensor_char.unsqueeze(0)
            #print(line_tensor_word.size(),line_tensor_char.size(),62)
            output,hidden = gru(line_tensor_char,None)
            guess, guess_i = categoryFromOutput(output)
            result.append({'Correct_ICD':category,'Predicted_ICD':guess,'MG_ID':MG_ID})
            #print(category,guess)         #uncomment this line for detailed label/prediction pairs
            cat_pred.append(guess)
            cat_true.append(category)
    print('----------------------------------------------')
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    print(f1score)
    writeToFile("f1score: " + str(f1score),out_text_filename)
    for i in range(len(result)):
        result[i] = str(result[i])
    writeToFile('\n'.join(result),out_results_filename)
    
    return

def test(model):
    tdata,all_categories = get_data(input_test)
    result = []
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 1000
    start = time.time()
    for k,v in tdata.iteritems():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            try:
                category, line, category_tensor, line_tensor_char = getTensors(k,v[i][1])
            except ValueError:
                print('----------------outsided text----------------')
                print(v[i][1])
                print('\t' in v[i][1])
                
                iter -= 1
                continue
            if line_tensor_char.size() == (0,):
                continue 
            line_tensor_word = getWordTensors(v[1])
            
            MG_ID = v[i][0]
            line_tensor_word = line_tensor_word.unsqueeze(0)
            line_tensor_char = line_tensor_char.unsqueeze(0)    #(batch=1,num,1) 
            #print(line_tensor_word.size(),line_tensor_char.size(),62)
            output,hidden = model(line_tensor_word,line_tensor_char,None)
            guess, guess_i = categoryFromOutput(output)
            result.append({'Correct_ICD':category,'Predicted_ICD':guess,'MG_ID':MG_ID})
            #print(category,guess)         #uncomment this line for detailed label/prediction pairs
            cat_pred.append(guess)
            cat_true.append(category)
    print('----------------------------------------------')
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    print(f1score)
    writeToFile("f1score: " + str(f1score),out_text_filename)
    for i in range(len(result)):
        result[i] = str(result[i])
    writeToFile('\n'.join(result),out_results_filename)
    
    return

if __name__ == '__main__':
    
    model = train_iter()
    #model = torch.load(out_model_filename)
    if choice == "gru":
        test_gru(model)
    if choice == "both":
        test(model)
    if choice == "cnn":
        test_cnn(model)
