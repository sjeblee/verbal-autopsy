
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
from sklearn import metrics
import numpy as np

# In[3]:


data={}
all_categories = []
input_train = '/u/yanzhaod/data/va/mds+rct/train_child_cat.xml'
out_model_filename = "./model_child_gru.pt"
tree = etree.parse(input_train)
for e in tree.iter("cghr_cat"):
        if e.text not in data:
             data[e.text]=[]
             all_categories.append(e.text)
count = 0
for e in tree.iter("narrative","cghr_cat"):
    count += 1
    if e.tag == "narrative":
        value= e.text
#         print(value)
        
    if e.tag == 'cghr_cat':
        data[e.text].append(value)
print('number of narratives: %d' %count)

# for k,v in data.iteritems():
#     print (k)
#     print ((u"\n".join(v)))

n_categories= len(all_categories)
  
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

print(letterToTensor('a'))
narr = data['1'][0]
print(lineToTensor(narr).size())


# In[79]:

import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 64
rnn = RNN(n_letters, n_hidden, n_categories)


# In[116]:

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size,15)
        self.lstm = nn.LSTM(15,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        input = self.encoder(input.long())
        output,hidden = self.lstm(input,hidden)
        output = self.linear(output[-1])
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 64
lstm = LSTM(n_letters, n_hidden, n_categories)


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

n_hidden = 64
emb_size = 15
gru = GRU(n_letters, n_hidden, n_categories,emb_size)

# In[104]:

#input = lineToTensor(narr)
#print(input.size())
#hidden = torch.zeros(1,1,n_hidden)

#output, next_hidden = lstm(input,None)
#print(output)
#print(output.size())


# In[105]:

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    #print(top_n)
    #print(top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

#print(categoryFromOutput(output))


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

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category,category_tensor, '/ line =', line)


# In[110]:

learning_rate = 0.005
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)


# In[113]:

# If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor, model_name):
#     hidden = rnn.initHidden()

   #lstm.zero_grad()
   #output, hidden = lstm(line_tensor,None)
   if model_name == 'gru':
        gru.zero_grad()
        output ,hidden = gru(line_tensor,None)
   elif model_name == 'lstm':
        lstm.zero_grad()
        output ,hidden = lstm(line_tensor,None)
   elif model_name == 'rnn':
        rnn.zero_grad()
        output ,hidden = rnn(line_tensor,None)
   loss = criterion(output, category_tensor)
   loss.backward()
   optimizer.step()

   return output, loss.item()

def save(model_name):
    if model_name == "lstm":
        torch.save(lstm, out_model_filename)
    elif model_name == "gru":
        torch.save(gru, out_model_filename)
    elif model_name == "rnn":
        torch.save(rnn, out_model_filename)
    print('Saved as %s' % out_model_filename)
   
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)

def train_iter(model_name):
    
    #n_iters = 1000
    n_iters = 0
    epochs = 10
    for k,v in data.iteritems():
        n_iters += epochs *len(v)
    print("n_iters = " + str(n_iters))
    print_every = 10
    plot_every = 100

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()
    cat_pred,cat_true = [],[]
    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor,model_name)

        current_loss += loss
        
        guess, guess_i = categoryFromOutput(output)
        #print('guess: %s;   category:  %s' %(guess, category))    
        cat_pred.append(guess)
        cat_true.append(category)
        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    #print(count/n_iters)
    f1score = metrics.f1score(np.asarray(cat_true),np.asarray(cat_pred),average="weighted")
    print("f1score: ")
    print(f1score)
    save(model_name)
    if model_name == 'gru':
        return gru
    elif model_name == 'lstm':
        return lstm
    elif model_name == 'rnn':
        return rnn
    print("Mistake")
    return None

def testTrainSet(model):
    total = 0;
    counter = 0;
    for k,v in data.iteritems():
            for i in range(len(v)):
                print(i)
                category, line, category_tensor, line_tensor = randomTrainingExample()
                output,hidden = model(line_tensor,None)
                guess, guess_i = categoryFromOutput(output)
                #print(guess,category)
                loss = criterion(output,category_tensor)
                loss.backward()
                optimizer.step()
                #print(guess,category)
                if guess == category:
                    counter += 1
            total += len(v)
        # Print iter number, loss, name and guess
    print(counter,total)
    acc = counter/total
    line = "average accuracy is: " + str(acc)
    print(line)
    writeToFile(line,out_file)
    return
# In[117]:

if __name__ == '__main__':
    
    model = train_iter('gru')
    #model = torch.load(out_model_filename)

    #testTrainSet(model)
