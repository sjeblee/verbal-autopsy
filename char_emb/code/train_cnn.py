
# coding: utf-8

# In[6]:

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
from sklearn import metrics
import numpy as np


# In[7]:
input_train = '/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml'
input_test = '/u/yanzhaod/data/va/mds+rct/test_adult_cat.xml'
out_file = "out_cnn_test.txt"           #output report text filename
out_model_filename = './model_cnn_adult.pt'   #output filename for the model
n_hidden = 64                           #hidden layer size
emb_size = 20                           #embedding size

data={}                #key: category index, value:list of narratives
seq_length = 0
all_categories = []  
tree = etree.parse(input_train)
for e in tree.iter("cghr_cat"):     #iterate each "node" with tag=cghr_cat 
        if e.text not in data:      #e.text: the cghr_cat value
             data[e.text]=[]
             all_categories.append(e.text)
             
for e in tree.iter("narrative","cghr_cat"):
    if e.tag == "narrative":        #get the content of narratives  
        value= e.text
        if(seq_length) < len(value):   #obtain the maximum sequence length through loop
            seq_length = len(value)
        if e.tag == 'cghr_cat':     
            data[e.text].append(value)
    if e.tag == 'cghr_cat':             #once the narrative is obtained, assign it to the corresponding cghr_cat
        try:            
                 data[e.text].append(value)
        except:
                 print('Warning: need to append a value but the value does not exist')


# for k,v in data.iteritems():
#     print (k)
#     print ((u"\n".join(v)))

n_categories= len(all_categories)

# In[8]:

all_text = ''
for v in data.itervalues():
    all_text = all_text + u"-".join(v)

vocab = set(all_text)   #get each of the unique characters
n_letters = len(vocab)

def letterToIndex(letter):
    try:
            return list(vocab).index(letter)
    except ValueError:
            print("Warning: the letter " + letter + "is not in the list")
            return 0

def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# Turn a line into a <line_length x 1>, where the second demension contains the category index
def lineToTensor(narrative):
    tensor = torch.zeros(seq_length,1)
    for li, letter in enumerate(narrative):
        tensor[li][0] = letterToIndex(letter)
    for i in range (1,seq_length-len(narrative)):
        tensor[len(narrative)+i][0]=0
    return tensor

narr = data['1'][0]

# In[9]:

import torch.nn as nn

class CNN(nn.Module):
    def __init__(self,sequence_length,input_size, hidden_size, output_size, emb_size):
        super(CNN, self).__init__()

        self.hidden_size = hidden_size
        self.seq_length = sequence_length
        self.encoder = nn.Embedding(input_size,emb_size)      # Embedding process
        self.conv2 = nn.Sequential(nn.Conv2d(1,3,kernel_size = (2,emb_size),stride=1), nn.ReLU())
        self.conv3 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(3,emb_size),stride=1),nn.ReLU())
        self.conv4 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(4,emb_size),stride=1),nn.ReLU())
        self.conv5 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(5,emb_size),stride=1),nn.ReLU())
        self.conv6 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(6,emb_size),stride=1),nn.ReLU())
        self.conv7 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(7,emb_size),stride=1),nn.ReLU())
        self.conv8 = nn.Sequential(nn.Conv2d(1,3,kernel_size=(8,emb_size),stride=1),nn.ReLU())
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


cnn = CNN(seq_length,n_letters, n_hidden, n_categories, emb_size)

# In[10]:
# the procedure to obtain output layer from the input narrative
input = lineToTensor(narr)
print('input size: ' + str(input.size()))
hidden = torch.zeros(1,1,n_hidden)
output= cnn(input)
print('output size: ' + str(output.size()))

# In[11]:
# Get the predicted category from the output layer (just get the maximum value)
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
#     print(top_n)
#     print(top_i.size())
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

#print(categoryFromOutput(output))


# In[12]:

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
    

#for i in range(10):
#    category, line, category_tensor, line_tensor = randomTrainingExample()
#    print('category =', category,category_tensor, '/ line =', line)


# In[27]: initiallization

learning_rate = 0.005
criterion = nn.NLLLoss()
optimizer = torch.optim.Adam(cnn.parameters(),lr=learning_rate)

# In[28]:

# If you set this too high, it might explode. If too low, it might not learn

def train(category_tensor, line_tensor):
#     hidden = rnn.initHidden()

   cnn.zero_grad()
   output = cnn(line_tensor)

   loss = criterion(output, category_tensor)
   loss.backward()
   optimizer.step()

   return output, loss.item()

def save():            #save the model to the .pt file (can reload it later!!)
   #save_filename = "./model_cnn.pt"
   torch.save(cnn, out_model_filename)
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


# In[ ]:

def trainIters(epochs):
    #print(data)
    #print(seq_length)
    print_every = 1000
    counter = 0
    current_loss = 0
    start = time.time()
    n_iters = 0
    TP = 0
    acc = []
    for k,v in data.iteritems():
            n_iters += epochs * len(v)
   
    for e in range(epochs):
        for k,v in data.iteritems():
            for i in range(len(v)):
                counter += 1
                category, line, category_tensor, line_tensor = getTensors(k,v[i])
                output, loss = train(category_tensor, line_tensor)
                current_loss += loss

                guess, guess_i = categoryFromOutput(output)
                #print(guess,category)
                if guess == category:
                        TP += 1
        # Print iter number, loss, name and guess
                if counter % print_every == 0:
                    guess, guess_i = categoryFromOutput(output)
                    correct = '✓' if guess == category else '✗ (%s)' % category
                    line = '%d %d%% (%s) %.4f %s / %s %s' % (counter, counter / n_iters * 100, timeSince(start), loss, line, guess, correct)
                    print(line)
                    writeToFile(line,"temp.txt")
        acc.append(TP/n_iters*epochs)
        TP = 0
    line = "accuracies over epochs" + str(acc)
    writeToFile(line,out_file)
    mean_acc = sum(acc)/epochs
    line = "average accuracy is: " + str(mean_acc)
    print(line)
    writeToFile(line,out_file)
    save()
    return cnn

def testTrainSet(cnn):
    total = 0;
    counter = 0;
    for k,v in data.iteritems():
            for i in range(len(v)):
                category, line, category_tensor, line_tensor = getTensors(k,v[i])
                output = cnn(line_tensor)
                guess, guess_i = categoryFromOutput(output)
                #print(guess,category)
                loss = criterion(output,category_tensor)
                loss.backward()
                optimizer.step()
                print(guess,category)
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


def test(cnn):
    Tdata={}
    tree = etree.parse(input_test)
    for e in tree.iter('cghr_cat'):
        if e.text not in Tdata:
            Tdata[e.text] = []
    for e in tree.iter("narrative","cghr_cat"):
        if e.tag == "narrative":
            value= e.text
            if seq_length < len(value):
                print("Warning: the length of the narr is longer than the sequence_length")
                print(seq_length,len(value))
                value = value[:seq_length]
        
        if e.tag == 'cghr_cat':
            #print(e.text,e.text in Tdata)
            Tdata[e.text].append(value)
            
    #cnn = torch.load('./model_cnn.pt')
    count = 0
    total = 0
    for k,v in Tdata.iteritems():
        for i in range(len(v)):
            category,line,category_tensor,line_tensor = getTensors(k,v[i])
            output = cnn(line_tensor)
            guess,guess_i = categoryFromOutput(output)
            if guess == category:
                count += 1
        total += len(v)
    try:
            line = "The accuracy on the test file is " + str(float(count)/float(total))
    except ZeroDivisionError:
            print("Error: Zero Devision Error, total is 0")
    print(line)
    writeToFile(line,out_file)

    cat_pred,cat_true = [],[]
    for k,v in Tdata.iteritems():
        for i in range(len(v)):
            category,line,category_tensor,line_tensor = getTensors(k,v[i])
            output = cnn(line_tensor)
            guess,guess_i = categoryFromOutput(output)
            cat_pred.append(guess)
            cat_true.append(category)

    print(cat_true,'true')
    print(cat_pred,'pred')
    print(len(cat_true),len(cat_pred))
    f1score = metrics.f1_score(np.asarray(cat_true),np.asarray(cat_pred),average="weighted")
    print("f1score: ")
    print(f1score)
    
    return     

# In[92]:

if __name__ == '__main__':
    
    #cnn = trainIters(10)
    cnn = torch.load(out_model_filename)
    testTrainSet(cnn)
    #test(cnn)
    


