
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


# In[3]:


data={}
all_categories = []
tree = etree.parse('../data/mds+rct/train_adult_cat.xml')
for e in tree.iter("cghr_cat"):
        if e.text not in data:
             data[e.text]=[]
             all_categories.append(e.text)
for e in tree.iter("narrative","cghr_cat"):
    if e.tag == "narrative":
        value= u''.join(e.text)
#         print(value)
        
    if e.tag == 'cghr_cat':
        data[e.text].append(value)


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

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)


# In[116]:

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()

        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size,20)
        self.lstm = nn.LSTM(20,hidden_size)
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

n_hidden = 128
lstm = LSTM(n_letters, n_hidden, n_categories)


# In[104]:

input = lineToTensor(narr)
print(input.size())
hidden = torch.zeros(1,1,n_hidden)

output, next_hidden = lstm(input,None)
print(output)
print(output.size())


# In[105]:

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    print(top_n)
    print(top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

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

for i in range(10):
    category, line, category_tensor, line_tensor = randomTrainingExample()
    print('category =', category,category_tensor, '/ line =', line)


# In[110]:

learning_rate = 0.005
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(lstm.parameters(),lr=learning_rate)


# In[113]:

# If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

   lstm.zero_grad()
   output, hidden = lstm(line_tensor,None)

   loss = criterion(output, category_tensor)
   loss.backward()
   optimizer.step()

   return output, loss.item()

def save():
   save_filename = "./model.pt"
   torch.save(lstm, save_filename)
   print('Saved as %s' % save_filename)
   
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)


# In[117]:

if __name__ == '__main__':
    
    n_iters = 100000
    print_every = 5000
    plot_every = 1000

    # Keep track of losses for plotting
    current_loss = 0
    all_losses = []
    start = time.time()

    for iter in range(1, n_iters + 1):
        category, line, category_tensor, line_tensor = randomTrainingExample()
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss

        # Print iter number, loss, name and guess
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct))

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    save()

