
# coding: utf-8

# In[78]:

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


# In[79]:


data={}
seq_length = 0
all_categories = []
tree = etree.parse('../data/mds+rct/train_adult_cat.xml')
for e in tree.iter("cghr_cat"):
        if e.text not in data:
             data[e.text]=[]
             all_categories.append(e.text)
for e in tree.iter("hindi_narrative","cghr_cat"):
    if e.tag == "hindi_narrative":
        value= u''.join(e.text)
        if(seq_length) < len(value):
            seq_length = len(value)
        
    if e.tag == 'cghr_cat':
        data[e.text].append(value)


# for k,v in data.iteritems():
#     print (k)
#     print ((u"\n".join(v)))

n_categories= len(all_categories)


# In[80]:

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
    tensor = torch.zeros(seq_length,1)
    for li, letter in enumerate(narrative):
        tensor[li][0] = letterToIndex(letter)
    for i in range (1,seq_length-len(narrative)):
        tensor[len(narrative)+i][0]=0
    return tensor

narr = data['1'][0]



# In[91]:

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,sequence_length,input_size, hidden_size, output_size):
        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.seq_length = sequence_length
        self.encoder = nn.Embedding(input_size,20)
        self.conv2 = nn.Sequential(nn.Conv2d(1,3,kernel_size = (2,20),stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(3,20),stride=1),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(4,20),stride=1),nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(5,20),stride=1),nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(6,20),stride=1),nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(7,20),stride=1),nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(8,20),stride=1),nn.ReLU())
        self.linear = nn.Linear(21,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input):
        input = self.encoder(input.long())
        input = input.squeeze(1)
#         print(input.unsqueeze(0).unsqueeze(0).size())
        output2 = self.conv2(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-2+1))
        output2 = self.maxpool(output2.squeeze(3))
#         print(output2.size())
        output3 = self.conv3(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-3+1))
        output3 = self.maxpool(output3.squeeze(3))
#         print(output3.size())
        output4 = self.conv4(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-4+1))
        output4 = self.maxpool(output4.squeeze(3))
#         print(output4.size())
        output5 = self.conv5(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-5+1))
        output5 = self.maxpool(output5.squeeze(3))
        
        output6 = self.conv6(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-6+1))
        output6 = self.maxpool(output6.squeeze(3))
        
        output7 = self.conv7(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-7+1))
        output7 = self.maxpool(output7.squeeze(3))
        
        output8 = self.conv8(input.unsqueeze(0).unsqueeze(0))
        self.maxpool = nn.MaxPool1d(kernel_size=(self.seq_length-8+1))
        output8 = self.maxpool(output8.squeeze(3))
        
        
        final_output = torch.cat((output2,output3,output4,output5,output6,output7,output8),1)
        final_output = self.linear(final_output.squeeze(2))
        final_output = self.softmax(final_output)
        return final_output

n_hidden = 64
cnn = CNN(seq_length,n_letters, n_hidden, n_categories)
    


# In[82]:

input = lineToTensor(narr)
print(input.size())
hidden = torch.zeros(1,1,n_hidden)

output= cnn(input)
print(output)
print(output.size())


# In[83]:

def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    print(top_n)
    print(top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

print(categoryFromOutput(output))


# In[84]:

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


# In[85]:

learning_rate = 0.005
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate)


# In[93]:

# If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

   cnn.zero_grad()
   output = cnn(line_tensor)

   loss = criterion(output, category_tensor)
   loss.backward()
   optimizer.step()

   return output, loss.item()

def save():
   save_filename = "./model_cnn_21x20.pt"
   torch.save(cnn, save_filename)
   print('Saved as %s' % save_filename)
   
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)

def writeToFile(line):
        if os.path.exists("output.txt"):
                append_write = 'a' # append if already exists
        else:
                append_write = 'w' # make a new file if not

        f = open("output.txt",append_write)
        f.write(line + '\n')
        f.close()
                                            
# In[92]:

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
            line = '%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line, guess, correct)
            print(line)
            writeToFile(line)

        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    save()


# In[ ]:



