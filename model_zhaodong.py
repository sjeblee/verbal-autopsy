import model_format
import sys

import numpy

import time

from keras.utils.np_utils import to_categorical
from sklearn import metrics

from sklearn import preprocessing

#import attention_utils
from model_lib_test import CNN_Text
import math
#from layers import Attention
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from lxml import etree
global anova_filter
global labelencoder
from allennlp.modules.elmo import Elmo, batch_to_ids
numpy.set_printoptions(threshold=numpy.inf)
import gc
import re
import utils
from lxml import etree
from word2vec import load
from word2vec import get
#from model_format import CNN_ELMO

global labelname
global cuda
global max_num_word
global rec_type
#global all_categories
labelencoder = preprocessing.LabelEncoder()
cuda = torch.device("cuda:0")
TRAINING = False        # To train the model; otherwise load existing model
USE_SERVER = False      # To run in CSLab server, otherwise home computer
#-----------parameters-------------
learning_rate = 0.003
num_epochs = 12
batch_size= 16
max_num_word = 200
emb_dim_char=24
max_char_in_word = 7
#-----------------------------------------------
emb_dim_word = 100
emb_dim_comb = max_char_in_word*emb_dim_char+emb_dim_word
stime = time.time()
rec_type = "child"
#---------------------------------
embedding = 'conc'
model_type = 'cnn'
modelfile = rec_type+'_'+embedding+'_'+model_type+'_model.pt'    # Filename of the saved model
if USE_SERVER:
    input_train = "/u/yanzhaod/data/va/mds+rct/train"+rec_type+"_cat_spell.xml"  #input train file for char_embeeding
    input_test = "/u/yanzhaod/data/va/mds+rct/test"+rec_type+"_cat_spell.xml"    #input test file for char_embedding
else:
    input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+rec_type+"_cat_spell.xml"
    input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+rec_type+"_cat_spell.xml"
def get_data(input_train):
    '''
    INPUT:
        input_train: path to the training xml file
    OUTPUT:
        data: a dictionary where keys are cghr_cat and values are lists of the input features
        all_categories: a list containing all the categories
    '''
    all_categories = []
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
def preprocess(input_train,embedding='elmo'):
    '''
    INPUT:
        input_train: path to the training xml file
    OUTPUT:
        X: a list of strings, containing the narratives
        ID: a list of strings, containing MG_IDs
        label: numpy array of string, containing the labels
    '''
    data,all_categories = get_data(input_train)
    if embedding == 'elmo':
        ID = []
        label = []
        sentence = []
        for k,v in data.items():
            for i in range(len(v)):
                ID.append(v[i][0])
                label.append(k)
                text = v[i][1].split()
                sentence.append(text)
        label = numpy.array(label)
        X = batch_to_ids(sentence)   
        del sentence
        
        
        if X.size(1) > max_num_word:
            X = X[:,:max_num_word,:]
        else:
            p2d = (0, 0, 0, max_num_word-X.size(1)) 
            X = F.pad(X, p2d, 'constant', 0)
        X = numpy.array(X)
    elif embedding == 'conc':
        char_fname = 'char_emb/code/char_emb_'+str(emb_dim_char)+'.txt'
        vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
        emb_dic = utils.get_dic(char_fname,vocab)
        wmodel,dim = load('D:/projects/zhaodong/research/va/data/dataset/narr+ice+medhelp.vectors.100')
        
        def letterToNumpy(letter):
            return numpy.array(emb_dic[letter])
        def lettersToNumpy(word):
            arr = numpy.zeros((max_char_in_word,emb_dim_char))
            for i in range(max_char_in_word):
                if i < len(word):
                    arr[i,:] = letterToNumpy(word[i])
            return arr
        def wordToNumpy(word):
            emb_word = numpy.array(get(word,wmodel))
            emb_letters = lettersToNumpy(word).flatten('C')    #'C' means to flatten in row major  #of shaoe(max_char_in_word*emb_dim_char,)
            return numpy.hstack((emb_word,emb_letters))    #(max_char_in_word*emb_dim_char+emb_dim_word,)
        def lineToNumpy(line):
            '''
            INPUT:
                line: string
            OUTPUT:
                emb_lin: numpy array of size (max_num_word,emb_dim_comb)
            '''
            l = line.split()
            emb_line = numpy.zeros((max_num_word,emb_dim_comb))
            for i in range(max_num_word):
                if i < len(l):
                    emb_line[i,:] = wordToNumpy(l[i])
            return emb_line
        ID = []
        X,label = [],[]
        for k,v in data.items():
            for i in range(len(v)):
                ID.append(v[i][0])
                X.append(lineToNumpy(v[i][1]))
                label.append(k)
        X = numpy.asarray(X).astype('float')
    return X,label


X,labels = preprocess(input_train,embedding=embedding)
testX,testlabels = preprocess(input_test,embedding=embedding)
labenc = labelencoder
all_labels = numpy.concatenate([labels,testlabels])
labenc.fit(all_labels)
Y = labenc.transform(labels).astype('int')
all_categories = list(set(list(all_labels)))
class_num = len(all_categories)
if TRAINING == True:
    if embedding == 'elmo':
        emb_dim = 1024
        model = model_format.CNN_ELMO(emb_dim,class_num,USE_SERVER=USE_SERVER)
        model.fit(X,Y,num_epochs=num_epochs,batch_size=batch_size,learning_rate=learning_rate)
    elif embedding == 'conc':
        model = model_format.CNN_Text(emb_dim_comb,class_num,USE_SERVER=USE_SERVER)
        model.fit(X,Y,num_epochs=num_epochs,batch_size=batch_size,learning_rate=learning_rate)
    torch.save(model, modelfile)
else:
    model = torch.load(modelfile)
y_pred = model.predict(testX)
labenc = labelencoder
predictedlabels = labenc.inverse_transform(y_pred)
f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
print('f1score: '+str(f1score))