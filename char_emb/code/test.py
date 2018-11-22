
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


# In[7]:


data={}  #every key of data is category index, value is a list of narratives
seq_length = 0
all_categories = []
tree = etree.parse('/u/yanzhaod/data/va/mds+rct/train_adult_cat.xml')
for e in tree.iter("cghr_cat"):   #get each content with tag=cghr_cat (here is a number)
        if e.text not in data:     
             data[e.text] = []
             all_categories.append(e.text)

for e in tree.iter("narrative","cghr_cat"):
        if e.tag == "narrative":               #get the content of narratives  
                value = e.text
                if (seq_length) < len(value):  #obtain the maximum sequence length through loop
                        seq_length = len(value)
                if e.tag == 'cghr_cat':
                        data[e.text].append(value)
        if e.tag == 'cghr_cat':                #once the narrative is obtained, assign it to the corresponding cghr_cat
                try:
                        data[e.text].append(value)
                except:
                        print('value not found')

#for k,v in data.iteritems():
#       print(k)
#       print(v)

all_text = ''
for v in data.itervalues():
        all_text += "-".join(v)

vocab = set(all_text)    #get each of the unique characters
n_letters = len(vocab)

def letterToIndex(letter):
        return list(vocab).index(letter)

def letterToTensor(letter):
        tensor = torch.zeros(1,n_letters)
        tensor[0][letterToIndex(letter)] = 1
        return tensor

# Turn a line into a <line_length x 1>, where the second demension contains the category index
# or an array of one-hot letter vectors
def lineToTensor(narrative):
        tensor = torch.zeros(seq_length,1)
        for li,letter in enumerate(narrative):        #convert the each letter of narrative to corresponding index
                tensor[li][0] = letterToIndex(letter)  
        for i in range(1,seq_length-len(narrative)):  #set the rest part to zero (redundant?)
                tensor[len(narrative)+i][0] = 0
        return tensor

narr = data['1'][0]

print(lineToTensor(narr).squeeze(1).unsqueeze(0).unsqueeze(0).squeeze(3).size())                
