
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
#input_test = '/u/yanzhaod/data/va/mds+rct/test_adult_cat.xml'
out_model_filename = "./char_emb/code/output/model_child_gru_128.pt"
out_text_filename = "char_emb/code/output/out_child_test_128.txt"
out_results_filename = 'char_emb/code/output/out_child_results.txt'

# Hidden size
n_hidden = 128            

# Embedding size
emb_dim_char =  30

# Learning rate
learning_rate = 0.0001


punctuation = ',;.!?:"_@#$%&*+-=<>()[]{}\''
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
            pattern = '[^\w\d\s<'+punctuation+'>]'
            text = re.sub(pattern,'',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
            text = re.sub('[\t\n]','',text)
            data[cghr_cat.text].append((MG_ID.text,text))
        except AttributeError:
            continue
    return data,all_categories
data,all_categories = get_data(input_train)
            
n_categories= len(all_categories)
vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '+punctuation
n_letters = len(vocab)
n_iters = 0
for k,v in data.iteritems():
    n_iters += len(v)
print("size of the narratives: %d" %n_iters)

print("vocab: %s" %vocab)

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
            #if True:              #just get a 
                emb = [float(i) for i in temp[1:]]
                emb_dic[letter] = emb 
    return emb_dic
emb_dic = get_dic(fname)
l = []
max_num_word = 0
max_num_char = 0

for k in data:
    v = data[k]
    for i in range(len(v)):
        if len(v[i][1]) > max_num_char:
            max_num_char = len(v[i][1])
            max_text = v[i][1]
        #word_list = re.split(' ',v[i][1])
        pattern = "[\w\d']+|["+punctuation+"]"
        word_list = re.findall(pattern, v[i][1])
        l.append((k,(v[i][0],word_list)))
        if len(word_list) > max_num_word:
            max_num_word = len(word_list)
shuffle(l)
print("max_num_word: %s; max_num_char:%s" %(max_num_word,max_num_char))
#print(len([i for i in max_text if i in punctuation]),11212)
def letterToTensor(letter):
    if letter not in emb_dic:
        print(letter,1)
    try:
        tensor = torch.tensor(emb_dic[letter],device=cuda)
    except KeyError:
        print("Warning: the letter " + letter + " is not in the dictionary")
        return torch.zeros([1,em_dim_char],device=cuda)  #better commeted out in training
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
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long,device=cuda)
    line_tensor = lineToTensor(line)
    category_tensor = category_tensor.to(cuda)
    return category, line, category_tensor, line_tensor

#max_num_word = 200   

wmodel,dim = load('/u/yanzhaod/data/narr_ice_medhelp.vectors.100')
emb_dim_word = len(get('have',wmodel))
class_num = n_categories
#(word_list)
#print(len(get(',',wmodel)))
# narr = data[data.keys()[0]][1][1]
# input = lineToTensor(narr)
# input=input.to(cuda)

class CNN_GRU_Text(nn.Module):
    def __init__(self, emb_dim_word,  emb_dim_char, class_num, hidden, max_num_char,kernel_num=200, kernel_sizes=5, dropout=0.0, ensemble=False, hidden_size=100):
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
        x = x.squeeze(1).squeeze(1).unsqueeze(0).unsqueeze(0) #1,1,200,100    
        x = F.relu(conv(x)).squeeze(3)  
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self,a,b, hidden):
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
        b = b.unsqueeze(1)
        b,hidden = self.gru(b,hidden)
        b = b.contiguous()
        b = b.view(-1,b.shape[2])
        b = self.linear(b)
        b = b.view(-1,b.size(0)*b.size(1))
        b = F.relu(b)
        
        #print(a.size(),b.size())   #(1,1000), (1,1000)
        input = torch.cat((a,b),1)      #1 is the horizontal concat
        output = self.fc1(input)
        output = self.softmax(output)
        return output, hidden

model = CNN_GRU_Text(emb_dim_word, emb_dim_char, n_categories, n_hidden,max_num_char)
model.to(cuda)

def getWordTensors(word_list):
    tensor = torch.zeros([max_num_word, emb_dim_word],device=cuda)
    for i in range(max_num_word):
        if i < len(word_list):
            tensor[i] = torch.tensor(get(word_list[i],wmodel),dtype=torch.long)
    tensor = torch.unsqueeze(tensor,1)   #d 
    return tensor

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
            
            #gru
            text = ' '.join(v[1])
            pattern = ' (?=['+punctuation+'])'
            text = re.sub(pattern, '', text)
            category, line, category_tensor, line_tensor_char = getTensors(k,text)
            if line_tensor_char.size() == (0,):
                continue  
            #cnn
            line_tensor_word = getWordTensors(v[1])
            line_tensor_char = line_tensor_char.to(cuda)
            #train
            optimizer.zero_grad()  
            output,hidden = model.forward(line_tensor_word,line_tensor_char,None)
            loss = criterion(output, category_tensor)
            loss.backward()
            optimizer.step()
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output)
                print('%d %d%% (%s) %.4f %s / %s' % (iter, iter / n_iters/epochs*100, timeSince(start), loss, line, guess))
            #guess, guess_i = categoryFromOutput(output)
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
                category, line, category_tensor, line_tensor_char = getTensors(k,v[i][1])
            except ValueError:
                print('----------------outsided text----------------')
                print(text)
                print('\t' in text)
                
                iter -= 1
                continue
            if line_tensor_char.size() == (0,):
                continue 
            line_tensor_word = getWordTensors(v[1])
            
            MG_ID = v[i][0]
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
    test(model)
