
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
data={}         
all_categories = []
input_train = '/u/yanzhaod/data/va/mds+rct/train_child_cat.xml'
input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat_spell.xml'
#input_test = '/u/yanzhaod/data/va/mds+rct/test_neonate_cat.xml'
out_model_filename = "./char_emb/code/output/model_child_cnngru_m2.pt"
out_text_filename = "char_emb/code/output/out_child_test_cnngru_m2.txt"
out_results_filename = 'char_emb/code/output/out_child_results.txt'

# Hidden size
n_hidden = 128            

# Embedding size
emb_dim_char =  30

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
            text = re.sub('[\t\n]','',text)
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

print(vocab)


fname = 'char_emb/code/char_emb_30.txt'
def get_dic(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [x.strip() for x in content]
    emb_dic = {}
    for i in range(len(content)):
        temp = content[i].split()
        if temp:
            letter = temp[0]
            if len(letter) > 4: #indication of a letter of space
                letter = ' '
                temp = [letter] + temp
            if letter in vocab:
                emb = [float(i) for i in temp[1:]]
                emb_dic[letter] = emb 
    return emb_dic
emb_dic = get_dic(fname)
print(' ' in emb_dic,'I lobv tat')
l = []
max_num_word = 0
for k in data:
    v = data[k]
    for i in range(len(v)):
        word_list = re.split(' ',v[i][1])
        l.append((k,(v[i][0],word_list)))
        if len(word_list) > max_num_word:
            max_num_word = len(word_list)
print(max_num_word,word_list)
max_num_word = 200   #________________
max_word_length = 10
shuffle(l)
wmodel,dim = load('/u/yanzhaod/data/narr_ice_medhelp.vectors.100')
emb_dim_word = len(get('have',wmodel))
emb_dim = max_word_length*emb_dim_char + emb_dim_word
class_num = n_categories
class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size,hidden_size)
        self.linear = nn.Linear(hidden_size*max_num_word,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        
        #input = self.encoder(input.long())
        #print(input.size(),1123323)
        output,hidden = self.gru(input,hidden)
        #print(output.size(),11221)
        output = self.linear(output.view(-1,output.size(0)*output.size(2)))
        output = self.softmax(output)
        
        return output, hidden
        
    def initHidden(self):
        return torch.zeros([1, self.hidden_size],device=cuda)


model = GRU(emb_dim, n_hidden, n_categories,)
model.to(cuda)


#!!!gru.to(cuda)
def letterToTensor(letter):
    tensor = torch.tensor(emb_dic[letter],device=cuda)
    return tensor

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

def wordToTensor(word):
    emb_word = torch.zeros([1,emb_dim_word],dtype=torch.long,device=cuda)
    emb_word[0] = torch.tensor(get(word,wmodel),dtype=torch.long,device=cuda)
    tensor_char = torch.zeros([max_word_length,emb_dim_char],dtype=torch.long,device=cuda)
    for i in range(max_word_length-1):
        if i < len(word):
            tensor_char[i] = letterToTensor(word[i])
    tensor_char[max_word_length-1] = letterToTensor(' ')
    #print(tensor_char.view(-1,emb_dim-emb_dim_word).size())
    #print(emb_word.size())
    return torch.cat((emb_word,tensor_char.view(-1,emb_dim-emb_dim_word)),1)
def getTensors(category,word_list):
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long,device=cuda)
    line_tensor = torch.zeros([max_num_word, emb_dim],device=cuda)
    for i in range(max_num_word):
        if i < len(word_list):
            line_tensor[i] = wordToTensor(word_list[i])
    line_tensor = torch.unsqueeze(line_tensor,1)
    return category,word_list,category_tensor, line_tensor
#print(getWordTensors('i have a dog'.split()).size(),1111)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def train(category_tensor, line_tensor):
    optimizer.zero_grad()  
    output,hidden = model.forward(line_tensor,None)
    loss = criterion(output, category_tensor)
    loss.backward()
    optimizer.step()
    return output, loss.item()

def save():
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


def train_iter():
    
    print_every = 1000
    plot_every = 5000
    epochs = 30
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    iter = 0
    for i in range(epochs):
        
        for e in l:
            iter += 1
            #print(iter)
            k,v = e[0],e[1]

            category, word_list, category_tensor, line_tensor = getTensors(k,v[1])
            if line_tensor.size() == (0,):
                continue  
            output, loss = train(category_tensor, line_tensor)
            current_loss += loss
            guess, guess_i = categoryFromOutput(output)
            if iter % print_every == 0:
                print('%d %d%% (%s) %.4f' % (iter, iter / n_iters/epochs*100, timeSince(start), loss))
    save()
    return model

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
                category, word_list, category_tensor, line_tensor = getTensors(k,v[i][1])
            except ValueError:
                print('----------------outsided text----------------')
                print(v[i][1])
                print('\t' in v[i][1])
                
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
    
    model = train_iter()
    # out_model_filename = 'output/model_adult_gru_128.pt'
    #model = torch.load(out_model_filename)
    # print(list(model.parameters())[0].data.numpy())
    test(model)