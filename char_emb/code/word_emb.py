
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


import subprocess
import time
import math 
from sklearn.metrics import f1_score
import numpy as np
from random import shuffle


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
cuda = torch.device("cuda:2")
data={}         
all_categories = []
input_train = '/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml'
#input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat_spell.xml'
input_test = '/u/yanzhaod/data/va/mds+rct/test_adult_cat.xml'
out_model_filename = "./char_emb/code/output/model_adult_gru_128.pt"
out_text_filename = "char_emb/code/output/out_adult_test_128.txt"
out_results_filename = 'char_emb/code/output/out_adult_results.txt'

# Hidden size
n_hidden = 128            

# Embedding size
emb_size =  30

# Learning rate
learning_rate = 0.0001

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
            text = re.sub('[^a-z0-9\s]','',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
            
            data[cghr_cat.text].append((MG_ID.text,text))
        except AttributeError:
            continue
    return data,all_categories
data,all_categories = get_data(input_train)
            
n_categories= len(all_categories)
vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
n_letters = len(vocab)
n_iters = 0
for k,v in data.iteritems():
    n_iters += len(v)
print("size of the narratives: %d" %n_iters)

def letterToIndex(letter):
    return vocab.index(letter)
def letterToTensor(letter):
    tensor = torch.zeros([1, n_letters],device=cuda)
    tensor[0][letterToIndex(letter)] = 1
    return tensor
def lineToTensor(narrative):
    tensor = torch.zeros([len(narrative),1],device=cuda)
    for li, letter in enumerate(narrative):

        tensor[li][0] = letterToIndex(letter)
    return tensor
narr = data[data.keys()[0]][1][1]
input = lineToTensor(narr)
input=input.to(cuda)


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
        #print(output.size())
        output = self.linear(output[-1])
        #print(output.size())
        
        
        
        
        output = self.softmax(output)
        
        return output, hidden
        
    def initHidden(self):
        return torch.zeros([1, self.hidden_size],device=cuda)

class CNN_Text(nn.Module):

    def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes=6, dropout=0.0, ensemble=False, hidden_size=100):
        super(CNN_Text, self).__init__()
        D = embed_dim
        C = class_num
        Ci = 1
        self.Co = kernel_num
        self.Ks = kernel_sizes
        self.ensemble = ensemble

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

gru = GRU(n_letters, n_hidden, n_categories,emb_size)
gru.to(cuda)
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    #print(top_n)
    #print(top_i.size())
    category_i = top_i[0].item()
    #print(category_i)
    return all_categories[category_i], category_i

def getTensors(category,line):
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long,device=cuda)
    line_tensor = lineToTensor(line)
    category_tensor = category_tensor.to(cuda)
    return category, line, category_tensor, line_tensor
    

criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(gru.parameters(),lr=learning_rate)


def train(category_tensor, line_tensor):
    optimizer.zero_grad()  
    
    output,hidden = gru.forward(line_tensor,None)

    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

def save():
    torch.save(gru, out_model_filename)
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

def train_iter():
    
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
                guess, guess_i = categoryFromOutput(output)
                correct = '✓' if guess == category else '✗ (%s)' % category
                print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters/epochs*100, timeSince(start), loss, line, guess, correct))
    
            # Add current loss avg to list of losses
            if iter % plot_every == 0:
                all_losses.append(current_loss / plot_every)
                current_loss = 0
    save()
    writeToFile("losses: " + str(all_losses),out_text_filename)
    return gru

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
    
   # model = train_iter()
    out_model_filename = 'output/model_adult_gru_128.pt'
    model = torch.load(out_model_filename)
    print(list(model.parameters())[0].data.numpy())