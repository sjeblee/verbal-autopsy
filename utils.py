
# coding: utf-8

# In[1]:

from __future__ import unicode_literals, print_function, division

import torch
import csv
import torch.nn as nn
import re
from io import open
import os

from lxml import etree
from torch.utils.data import DataLoader
import torch.nn.functional as F
import time
import math 
import numpy as np
import data_util
from gensim.models import KeyedVectors, Word2Vec
from sklearn import metrics
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
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

def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)

def get(word, model):
    dim = model.vector_size
    if word in model: #.wv.vocab:
        return list(model[word])
    else:
        return data_util.zero_vec(dim)

def load(filename):
    if '.bin' in filename:
        model = load_bin_vectors(filename, True)
    elif 'fasttext' in filename:
        model = FastText.load(filename)
    elif '.wtv' in filename:
        model = Word2Vec.load(filename)
    else:
        model = load_bin_vectors(filename, False)
    dim = model.vector_size
    return model, dim
def load_bin_vectors(filename, bin_vecs=True):
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=bin_vecs, unicode_errors='ignore')
    return word_vectors

def save_numpy(variables,filenames):
    '''input:
        variables: a list of numpy arrays
        filenames: list of paths corresponds to the path of the np arrays
        variables and filenames shares the same length
    '''
    for i in range(len(variables)):
        np.save(filenames[i],variables[i])
    return

def load_numpy(filenames):
    '''
    load the numpy arrays in a tuple given filenames
    '''
    l = []
    for file in filenames:
        l.append(np.load(file))
    return tuple(l) 

def stats_from_results(testlabels,predictedlabels,testids,PRINT=True):
    '''
    input:
        testlabels: numpy array for the given labels
        predictedlabels: numpy array for predicted labels
        testids:numpy array for MG_IDs
    output
        precision: float
        recall:float
        f1score float
        csmf_accuracy: float
    '''
    f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
    precision = metrics.precision_score(testlabels, predictedlabels, average='weighted')
    recall = metrics.recall_score(testlabels, predictedlabels, average='weighted')
    result  =[]
    for j in range(len(testids)):
        result.append({'Correct_ICD':testlabels[j],'Predicted_ICD':predictedlabels[j],'MG_ID':testids[j]})
    csmf_accuracy = get_csmf_acc(result)
    if PRINT:
        print('precision: '+str(precision))
        print('recall: '+str(recall))
        print('f1score: '+str(f1score))
        print('csmf_accuracy: '+str(csmf_accuracy))
    return precision, recall, f1score, csmf_accuracy

def get_csmf_acc(result):
    '''
    input:
        result: a list of dictionaries in the format 
                    {'Correct_ICD':testlabels[j],'Predicted_ICD':predictedlabels[j],'MG_ID':testids[j]}
    output:
        csmf_accuracy:a float for csmf accuracy
    '''
    labels_correct = {}
    labels_pred = {}
    correct = []
    predicted = []
    for res in result:
        pred = res['Predicted_ICD']
        predicted.append(pred)
        cor = res['Correct_ICD']
        correct.append(cor)
#            print(labels_correct.keys())
        if cor in labels_correct:
            labels_correct[cor] = labels_correct[cor] + 1
        else:
            labels_correct[cor] = 1

        if pred in labels_pred:
            labels_pred[pred] = labels_pred[pred] + 1
        else:
            labels_pred[pred] = 1
    n = len(correct)
    csmf_pred = {}
    csmf_corr = {}
    csmf_corr_min = 1
    csmf_sum = 0
    for key in labels_correct.keys():
        if key not in labels_pred:
            labels_pred[key] = 0
        num_corr = labels_correct[key]
        num_pred = labels_pred[key]
        csmf_c = num_corr/n
        csmf_p = num_pred/n
        csmf_corr[key] = csmf_c
        csmf_pred[key] = csmf_p
        #print "csmf for " + key + " corr: " + str(csmf_c) + ", pred: " + str(csmf_p)
        if csmf_c < csmf_corr_min:
            csmf_corr_min = csmf_c
        csmf_sum = csmf_sum + abs(csmf_c - csmf_p)

    csmf_accuracy = 1 - (csmf_sum / (2 * (1 - csmf_corr_min)))
    return csmf_accuracy

def get_dic(fname,vocab):
    '''
    fname: filenames of the character embeddings
    vocab: a string
    '''
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

def get_data_phmrc(input_train,label_dic):
    data={} 
    tree = etree.parse(input_train)
    root = tree.getroot()
    for child in root:
        MG_ID = child.find('MG_ID')
        narrative = child.find('narrative')
        phmrc_cat = child.find('cat_phmrc')
        text = narrative.text.lower()
        text = re.sub('[^a-z0-9 ]','',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
        text = re.sub('[\t\n]','',text)
        text = re.sub(' +', ' ', text)
        label_text = label_dic[phmrc_cat.text]
        if label_text in data:
            data[label_text].append((MG_ID.text,text))
        else:
            data[label_text]=[(MG_ID.text,text)]
    all_categories = list(set(data.keys()))
    return data,all_categories

#     
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
def combineFeat(X2,testX2,feat):
    from keras.utils.np_utils import to_categorical
    '''
    X2 and testX2 are the numpy array with type float
    '''
    if type(feat) == list:
        i = 0 
        limit = 0
        while i < len(feat):
#            print(X2[:,i],limit,100101)
            X2[:,i] = np.add(X2[:,i],limit)
            testX2[:,i] = np.add(testX2[:,i],limit)
            combinedX2 = np.concatenate([X2[:,i],testX2[:,i]])
            limit = np.amax(combinedX2)
            i += 1
        X2 = X2.flatten('F')
        testX2 = testX2.flatten('F')
        return combineFeat(X2,testX2,'singlefeat')
            
    else:
        lenX2 = X2.shape[0]
        combinedX2 = np.concatenate([X2,testX2])
        combinedX2 = to_categorical(combinedX2)
        return combinedX2[:lenX2,:],combinedX2[lenX2:,:]
def get_data_struct(input_train,feature):
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
        if type(feature) == list:
            dic = {}
            for f in feature:
                try:
                    dic[f] = child.find(f).text
                except AttributeError:
                    dic[f] = None
                
        else:
            try:
                feature_val = child.find(feature).text
            except AttributeError:
                feature_val = None
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
        if type(feature) == list:
            data[cghr_cat.text].append((MG_ID.text,text,dic))
        else:
            data[cghr_cat.text].append((MG_ID.text,text,feature_val))
    return data,all_categories



