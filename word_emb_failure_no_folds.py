
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
from torch.utils.data import DataLoader
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
        
all_categories = []
input_train = "D:/projects/zhaodong/research/va/data/dataset/train_adult_cat_spell.xml"
#input_test = '/u/yanzhaod/data/va/mds+rct/test_child_cat_spell.xml'
input_test = "D:/projects/zhaodong/research/va/data/dataset/test_adult_cat_spell.xml"
out_model_filename = "./char_emb/code/output/model_adult_gru_128.pt"
out_text_filename = "char_emb/code/output/out_adult_test_128.txt"
out_results_filename = 'char_emb/code/output/out_adult_results.txt'

# Hidden size
n_hidden = 128            

# Embedding size
emb_dim_char =  30

# Learning rate
learning_rate = 0.0001

stime = time.time()
print("preprocessing...")
def get_data(input_train):
    data={} 
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
        cghr_cat2 = [child.find('CODINGKEYWORDS1'),child.find('CODINGKEYWORDS2')]
        second_try = []
        try:
            text = narrative.text.lower()
        except AttributeError:
            for e in cghr_cat2:
                try:
                    second_try.append(child.find('CODINGKEYWORDS1').text.lower())
                except AttributeError:
                    continue
            if len(second_try) == 2:
                if second_try[0] == second_try[1]:
                    second_try = second_try[0]
                else:
                    second_try = ' '.join(second_try)
            elif len(second_try) == 1:
                second_try = second_try[0]
            else:
                print("undetected mgid: "+MG_ID.text)
        if second_try:
            text = text + ' ' + second_try.lower()
            #print(MG_ID.text)
        text = re.sub('[^a-z0-9 ]','',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
        text = re.sub('[\t\n]','',text)
        text = re.sub(' +', ' ', text)
        data[cghr_cat.text].append((MG_ID.text,text))
    return data,all_categories
data,all_categories = get_data(input_train)
            
n_categories= len(all_categories)
vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
print("vocab: %s" %vocab)
n_letters = len(vocab)
n_iters = 0
for k,v in data.items():
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
print("max number of word %d" %max_num_word)
print("max number of char %d" %max_num_char)

def letterToTensor(letter):
    tensor = torch.tensor(emb_dic[letter],device=cuda)
    return tensor
def lineToTensor(narrative):
    tensor = torch.zeros([max_num_char,emb_dim_char],device=cuda)
    for li, letter in enumerate(narrative):
        try:
            tensor[li] = letterToTensor(letter)
        except IndexError: #for test set, it's length may exceed max_num_char
            break
    return tensor
# narr = data[data.keys()[0]][1][1]
# input = lineToTensor(narr)
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
    
#max_num_word = 200   
shuffle(l)
wmodel,dim = load('D:/projects/zhaodong/research/va/data/dataset/narr+ice+medhelp.vectors.100')
emb_dim_word = len(get('have',wmodel))
class_num = n_categories


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
        #print(a.size(),b.size(),51) #[128, 200, 1, 100],[128, 1484, 30]
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

model = CNN_GRU_Text(emb_dim_word, emb_dim_char, n_categories, n_hidden)
model.to(cuda)

def getWordTensors(word_list):
    tensor = torch.zeros([max_num_word, emb_dim_word],device=cuda)
    for i in range(max_num_word):
        if i < len(word_list):
            tensor[i] = torch.tensor(get(word_list[i],wmodel),dtype=torch.float)
    tensor = torch.unsqueeze(tensor,1)   #d 
    return tensor
    
#print(getWordTensors('i have a dog'.split()).size(),1111)
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
def train(category_tensor, line_tensor):
    optimizer.zero_grad()  
    output,hidden = gru.forward(line_tensor,None)
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
    
    batch_size = 16
    training_set = []
    for e in l:
        k,v = e[0],e[1]
        
        #gru
        category, line, category_tensor, line_tensor_char = getTensors(k,' '.join(v[1]))
        if line_tensor_char.size() == (0,):
            continue  
        #cnn
        line_tensor_word = getWordTensors(v[1])
        
        line_tensor_char = line_tensor_char.to(cuda)  
        training_set.append([line_tensor_word,line_tensor_char,category_tensor])
    training_params = {"batch_size": batch_size,
                "shuffle": True,
                "num_workers": 0}
    training_generator = DataLoader(training_set, **training_params)
    print("Preprocessing took %s"%timeSince(stime))
    print("Training...")
    count = 0
    for i in range(epochs):
        for iter, batch in enumerate(training_generator):
            count += 1
            line_tensor_word, line_tensor_char, category_tensor= batch
            #train
            optimizer.zero_grad()  
            output,hidden = model.forward(line_tensor_word,line_tensor_char,None)
            #print(output.size(),category_tensor.size(),2)  ((batch,9);(batch,))
#            print(output.size(),1)
#            print(category_tensor.size(),2)
            loss = criterion(output, category_tensor)
            loss.backward()
            optimizer.step()
            if iter % print_every == 0:
                guess, guess_i = categoryFromOutput(output)
                print('%d %d%% (%s) %.4f %s / %s' % (iter, iter / n_iters/epochs*100, timeSince(start), loss, line, guess))
    print("Training took %s"%timeSince(stime))
    save()
    return model
def test(model):
    tdata,all_categories = get_data(input_test)
    lt = []
    for k in tdata:
        v = tdata[k]
        for i in range(len(v)):
            word_list = re.split(' ',v[i][1])
            lt.append((k,(v[i][0],word_list)))
    print("size of the testing set %d"%len(lt))
    testing_set = []
    for e in lt:
        k,v = e[0],e[1]
        
        #gru
        category, line, category_tensor, line_tensor_char = getTensors(k,' '.join(v[1]))
        if line_tensor_char.size() == (0,):
            print("empty line tensor")
            continue  
        #cnn
        line_tensor_word = getWordTensors(v[1])
        
        line_tensor_char = line_tensor_char.to(cuda)  
        testing_set.append([line_tensor_word,line_tensor_char,category_tensor,category])
    batch_size = 16
    testing_params = {"batch_size": batch_size,
                       "shuffle": True,
                       "num_workers": 0}
    testing_generator = DataLoader(testing_set, **testing_params)
    cat_pred,cat_true = [],[]
    for iter, batch in enumerate(testing_generator):
        line_tensor_word, line_tensor_char, category_tensor,category= batch 
        output,hidden = model.forward(line_tensor_word,line_tensor_char,None)
        l = []
        for i in range(output.size(0)):
            guess,guess_i = categoryFromOutput(output[i])
            l.append(guess)
#        guess, guess_i = categoryFromOutput(output)
        guess=l
        cat_pred += list(guess)
        cat_true += category
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    return f1score
def test_backup(model):
    tdata,all_categories = get_data(input_test)
    
    result = []
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 1000
    start = time.time()
    for k,v in tdata.items():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            try:
                category, line, category_tensor, line_tensor_char = getTensors(k,' '.join(v[i][1]))
                
            except ValueError:
                print('----------------outsided text----------------')
                print(v[i][1])
                print('\t' in v[i][1])
                
                iter -= 1
                continue
            if line_tensor_char.size() == (0,):
                continue 
            line_tensor_word = getWordTensors(v[i][1])
            MG_ID = v[i][0]
            line_tensor_word = line_tensor_word.unsqueeze(0)
            line_tensor_char = line_tensor_char.unsqueeze(0)    #(batch=1,num,1) 
            #print(line_tensor_word.size(),line_tensor_char.size(),62)
            print(line_tensor_word)
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
    f1_score = test(model)
    print('f1score: '+str(f1_score))
