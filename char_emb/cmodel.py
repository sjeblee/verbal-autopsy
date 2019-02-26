
# coding: utf-8

# In[1]:

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

import math 
import time
import math
from sklearn.metrics import f1_score
import numpy as np
from random import shuffle
import torch.nn as nn

###########################################################
#**********************Parameters************************
#cuda -> the selected gpu
#input_train -> (string) the path of the input training file
#out_model_filename -> (string) the stored net file that can be reloaded lator
#out_text_filename -> (string) the output text file showing some relevant info
#out_results_filename -> (string) the output file for buiding the confusion matrix 
#n_hidden -> (int) hidden size of the neural net, usually 64 or 128
#emb_size -> (int) embedding size
#learning_rate -> (float) learning rate
#**********************Variables************************
#data  ->   (dictionary), key=(string) cghr category; value=(tuple) ((string) MG_ID, (string)narrative text) 
#vocab  ->  (string list), contains each of the letters
#n_letters -> (int), number of letters
###########################################################

# In[3]:
# cuda = torch.device("cuda:0")
# data={}         
# all_categories = []
# input_train = '/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml'
# #input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat_spell.xml'
# input_test = '/u/yanzhaod/data/va/mds+rct/test_adult_cat.xml'
# out_model_filename = "./code/output/model_adult_gru_128.pt"
# out_text_filename = "code/output/out_adult_test_128.txt"
# out_results_filename = 'code/output/out_adult_results.txt'
# 
# # Hidden size
# n_hidden = 128            
# 
# # Embedding size
# emb_size =  30
# 
# # Learning rate
# learning_rate = 0.0001

def get_data(input_train):
    tree = etree.parse(input_train)
    for e in tree.iter("cghr_cat"):
            text = e.text.lower()
            if text not in data:
                data[text]=[]
                all_categories.append(text)
                
    
    root = tree.getroot()
    for child in root:
        MG_ID = child.find('MG_ID')
        narrative = child.find('narrative')
        cghr_cat = child.find('cghr_cat')
        try:
            text = narrative.text.lower()
            #text = re.sub('[^a-z0-9\s\!\@\#\$\%\^\&\*\(\)\.\,]','',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
            text = re.sub('[^a-z0-9\s]','',text)
            text = re.sub(' +',' ',text)
            
            data[cghr_cat.text].append((MG_ID.text,text))
        except AttributeError:
            continue
    return data, all_categories
    
# data,all_categories = get_data(input_train)
# n_categories= len(all_categories)
# n_iters = 0
# vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
# n_letters = len(vocab)

# for k,v in data.iteritems():
#     n_iters += len(v)
# print("size of the narratives: %d" %n_iters)

def letterToIndex(letter):
    return vocab.index(letter)

def letterToTensor(letter):
    tensor = torch.zeros([1, n_letters],device=cuda)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(narrative):
    tensor = torch.zeros([len(narrative),1],device=cuda)
    for li, letter in enumerate(narrative):

        tensor[li][0] = letterToIndex(letter)
    return tensor

# narr = data[data.keys()[0]][1][1]
# input = lineToTensor(narr)
# input=input.to(cuda)

class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size,emb_size)
        self.gru = nn.GRU(emb_size,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = self.encoder(input.long())
        output,hidden = self.gru(input,hidden)
        output = self.linear(output[-1])
        output = self.softmax(output)
        return output, hidden
        
    def initHidden(self):
        return torch.zeros([1, self.hidden_size],device=cuda)

# gru = GRU(n_letters, n_hidden, n_categories,emb_size)
# gru.to(cuda)

# In[105]:
def categoryFromOutput(output,all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def getTensors(category,line):
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long,device=cuda)
    line_tensor = lineToTensor(line)
    category_tensor = category_tensor.to(cuda)
    return category, line, category_tensor, line_tensor
    
# criterion = nn.NLLLoss()
# optimizer = torch.optim.Adam(gru.parameters(),lr=learning_rate)

def train(category_tensor, line_tensor):
    optimizer.zero_grad()  
    output,hidden = gru(line_tensor,None)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

def save(model,out_model_filename):
    torch.save(model, out_model_filename)
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

def train_iter(gru,all_categories):
    print_every = 1000
    plot_every = 5000
    epochs = 30
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    iter = 0
    #for iter in range(1, n_iters + 1):
    l = []
    for k in data:
        v = data[k]
        for i in range(len(v)):
            l.append((k,v[i]))
    shuffle(l)
    print(len(l))
    for i in range(epochs):

        for e in l:
            k,v = e[0],e[1]
            iter += 1
            category, line, category_tensor, line_tensor = getTensors(k,v[1])
            if line_tensor.size() == (0,):
                continue  
            output, loss = train(category_tensor, line_tensor)
            current_loss += loss
            guess, guess_i = categoryFromOutput(output)
            #print('guess: %s;   category:  %s' %(guess, category))    
            # Print iter number, loss, name and guess
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output,all_categories)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters/epochs*100, timeSince(start), loss, line, guess, correct))
    
            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
    save()
    writeToFile("losses: " + str(all_losses),out_text_filename)
    return model


def testTrainSet(model):
    result = []
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 1000
    start = time.time()
    for k,v in data.iteritems():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            category, line, category_tensor, line_tensor = getTensors(k,v[i][1])
            if line_tensor.size() == (0,):
                continue 
            MG_ID = v[i][0]
            output,hidden = model(line_tensor,None)
            guess, guess_i = categoryFromOutput(output)
            result.append({'Correct_ICD':category,'Predicted_ICD':guess,'MG_ID':MG_ID})
            #print(category,guess)         #uncomment this line for detailed label/prediction pairs
            cat_pred.append(guess)
            cat_true.append(category)
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    print(f1score)
    writeToFile("f1score: " + str(f1score),out_text_filename)
    for i in range(len(result)):
        result[i] = str(result[i])
    writeToFile('\n'.join(result),out_results_filename)
    
    return
def test(model):
    print(input_test)
    
    for k in data:
        data[k] = []
    tree = etree.parse(input_test)
    root = tree.getroot()
    
    for child in root:
        MG_ID = child.find('MG_ID')
        narrative = child.find('narrative')
        cghr_cat = child.find('cghr_cat')
        try:
            text = narrative.text.lower()
            text = re.sub('[^a-z0-9\s]','',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
            text = re.sub(r'(^[ \t]+|[ \t]+(?=:))', '', text, flags=re.M)
            data[cghr_cat.text].append((MG_ID.text,text))
        except AttributeError:
            continue


    result = []
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 1000
    start = time.time()
    for k,v in data.iteritems():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            try:
                category, line, category_tensor, line_tensor = getTensors(k,v[i][1])
            except ValueError:
                print('----------------outsided text----------------')
                print(text)
                print('\t' in text)
                
                iter -= 1
                continue
            if line_tensor.size() == (0,):
                continue 
            MG_ID = v[i][0]
            output,hidden = model(line_tensor,None)
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
    print('finished')
    #model = train_iter()
    #model = torch.load(out_model_filename)

   ##   testTrainSet(model)
    # test(model)