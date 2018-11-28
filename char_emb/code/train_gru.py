
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
import time
import math 
import time
import math
from sklearn.metrics import f1_score
import numpy as np

# In[3]:

###########################################################
#**********************Variables************************
#data  ->   (dictionary), key=(string) cghr category; value=(string) narrative text 
#vocab  ->  (string list), contains each of the letters
#n_letters -> (int), number of letters

###########################################################



data={}         
all_categories = []
input_train = '/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml'
out_model_filename = "./output/model_adult_gru2.pt"
out_text_filename = "output/out_gru_test.txt"
tree = etree.parse(input_train)
for e in tree.iter("cghr_cat"):
        if e.text not in data:
             data[e.text]=[]
             all_categories.append(e.text)
count = 0
narr = True


#sometimes like in "MG_ID" = c1247, there is no narratives
l = []
ID = 'null'
counter = 200
LAST = False
for e in tree.iter("narrative","cghr_cat"):
    # if e.tag == "MG_ID":
    #     ID = e.text
    #if counter > 1:
    #     print(e.tag,counter,ID)
    # counter -= 1
    
    if LAST == False:
        LAST = e.tag
    else:
        if LAST == e.tag and e.tag == 'cghr_cat':
            continue   #ignore the case where cghr_cat appears two times
        if LAST != e.tag:  #desired case
            LAST = e.tag
    if e.tag == "narrative":
        value= e.text
        count += 1
    if e.tag == 'cghr_cat':
        data[e.text].append(value)
        
    
print('number of narratives: %d' %count)

print("keys of data:")
print(data.keys())
# print("one value of data:")
# example_key = data.keys()[0]
# print(data[example_key])

n_categories= len(all_categories)

n_iters = 0

for k,v in data.iteritems():
    n_iters += len(v)
print("size of the narratives: %d" %n_iters)
# In[78]:

all_text = ''
for v in data.itervalues():
    all_text = all_text + u"-".join(v)

vocab = set(all_text)            
n_letters = len(vocab)

def letterToIndex(letter):
    return list(vocab).index(letter)

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1 x n_letters>,
# or an array of one-hot letter vectors
def lineToTensor(narrative):
    tensor = torch.zeros(len(narrative),1)
    for li, letter in enumerate(narrative):
        tensor[li][0] = letterToIndex(letter)
    return tensor

#
print("a sample tensor of letter 'a':")
print(letterToTensor('a'))
narr = data['1'][0]
print("the size of a sample input (narr): ")
input = lineToTensor(narr)
print(input.size())


# In[79]:

import torch.nn as nn

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
        return torch.zeros(1, self.hidden_size)

n_hidden = 32
emb_size = 10
gru = GRU(n_letters, n_hidden, n_categories,emb_size)

# In[104]:


hidden = torch.zeros(1,1,n_hidden)

output, hn = gru(input,None)
print("the sample output and output size: ")
print(output,output.size())
print(output[-1])
print(hn.view(hn.size()[1], hn.size(2)))

# In[105]:

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    #print(top_n)
    #print(top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print("category from output:")
print(categoryFromOutput(output))


# In[109]:

import random
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample():
    category = randomChoice(all_categories)
    line = randomChoice(data[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
def getTensors(category,line):
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
    
print("sample category,line,category_tensor from the randomTraningExample")
#line_tensor should look like tensor([[77.],[88.],])
category, line, category_tensor, line_tensor = randomTrainingExample()
print('category =', category,category_tensor, '/ line =', line)


# In[110]:

learning_rate = 0.002
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(gru.parameters(),lr=learning_rate)


# In[113]:

# If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
    optimizer.zero_grad()      
    output,hidden = gru(line_tensor,None)

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
    
    print_every = 100
    plot_every = 500
    epochs = 7
    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    iter = 0
    #for iter in range(1, n_iters + 1):
    for e in range(epochs):
        for k,v in data.iteritems():
            for i in range(len(v)):
                iter += 1
                category, line, category_tensor, line_tensor = getTensors(k,v[i])
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


def testTrainSet(model):
    cat_pred,cat_true = [],[]
    iter = 0
    print_every = 100
    start = time.time()
    for k,v in data.iteritems():
        for i in range(len(v)):
            iter += 1
            if iter % print_every == 0:
                print(iter,timeSince(start))
            category, line, category_tensor, line_tensor = getTensors(k,v[i])
            output,hidden = gru(line_tensor,None)
            guess, guess_i = categoryFromOutput(output)
            cat_pred.append(guess)
            cat_true.append(category)
            print(guess,category)
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    print(f1score)
    writeToFile("f1score: " + str(f1score),out_text_filename)
    return
# In[117]:

if __name__ == '__main__':
    
    model = train_iter()
    #model = torch.load(out_model_filename)

    testTrainSet(model)
