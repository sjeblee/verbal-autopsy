# -*- coding: utf-8 -*-
"""
Created on Sun May  5 08:10:00 2019

@author: Zhaodong Yan
"""
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
labelencoder = None
numpy.set_printoptions(threshold=numpy.inf)
import gc
import re
from lxml import etree
def main():
    global labelname
    global cuda
    global labelencoder
    global num_word
    global rec_type
    labelencoder = preprocessing.LabelEncoder()
    cuda = torch.device("cuda:0")
    train_mode = True
    #-----------parameters-------------
    learning_rate = 0.003
    num_epochs = 12
    batch_size= 16
    num_word = 200
    stime = time.time()
    rec_type = "adult"
    #---------------------------------
    modelfile = rec_type+"_model.pt"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#    weight_file = 'D:/projects/zhaodong/research/elmo_pubMed_only.hdf5'
    weight_file = '/u/yanzhaod/data/elmo_pubMed_only.hdf5'
    elmo = Elmo(options_file, weight_file, 1, dropout=0)
    elmo = elmo.to(cuda)
    

    f1scores = []
    precisions = []
    recalls = []
    csmf = []
    #--------------------Build model and train-----------------
    for i in range(10):
        input_train = "/u/yanzhaod/data/small_dataset/train_"+rec_type+"_"+str(i)+"_cat_spell.xml"  #input train file for char_embeeding
        input_test = "/u/yanzhaod/data/small_dataset/test_"+rec_type+"_"+str(i)+"_cat_spell.xml"    #input test file for char_embedding
        if train_mode== True:
            X,Y,label,ID = preprocess(input_train)
            print("X: " + str(X.size()) +" Y: " + str(Y.shape))
            model = cnn_model(X,Y,elmo,batch_size=batch_size,num_epochs=num_epochs,learning_rate=learning_rate)
            torch.save(model, modelfile)
        else:
            model = torch.load(modelfile)
        etime = time.time()
        print("training took " + str(etime - stime) + " s")
        testids, testlabels, predictedlabels = test(model,input_test,elmo)
        print("Real Labels shape: " + str(testlabels))
        print("Predicted Labels shape: " + str(predictedlabels))
    
        f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
        precision = metrics.precision_score(testlabels, predictedlabels, average='weighted')
        result  =[]
        for j in range(len(testids)):
            result.append({'Correct_ICD':testlabels[j],'Predicted_ICD':predictedlabels[j],'MG_ID':testids[j]})
        recall = metrics.recall_score(testlabels, predictedlabels, average='weighted')
        csmf_accuracy = stats_from_result(result)
        print('precision: '+str(precision))
        print('recall: '+str(recall))
        print('f1score: '+str(f1score))
        print('csmf_accuracy: '+str(csmf_accuracy))
        f1scores.append(f1score)
        precisions.append(precision)
        recalls.append(recall)
        csmf.append(csmf_accuracy)
def preprocess(input_file):        
    data,all_categories = get_data(input_file)
    ID = []
    label = []
    sentence = []
    for k,v in data.items():
        for i in range(len(v)):
            ID.append(v[i][0])
            label.append(k)
            text = v[i][1].split()
            sentence.append(text)
    gc.collect()
    del data
    X = batch_to_ids(sentence)   
    del sentence
    
    
    if X.size(1) > num_word:
        X = X[:,:num_word,:]
    else:
        p2d = (0, 0, 0, num_word-X.size(1)) 
        X = F.pad(X, p2d, 'constant', 0)
        
    X = X.to(cuda)
    X.contiguous()
    labenc = labelencoder
    label = numpy.array(label)
    real_label = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '2', '3', '4', '5', '6', '7', '8', '9']
    if rec_type == 'adult':
        
        labenc.fit(real_label)
    else:
        labenc.fit(label)
    Y = labenc.transform(label)
    Y = to_categorical(Y)
    return X,Y,label,ID
def stats_from_result(result):
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
def cnn_model(X,Y,elmo,batch_size=16,num_epochs=12,learning_rate=0.001):
    Y = Y.astype('int') 
    X_len = X.size(0)
    dim = 1024
    num_epochs = num_epochs
    num_labels = Y.shape[-1]
    steps = 0
    learning_rate = 0.001
    cnn = CNN_Text(dim, num_labels)
    cnn = cnn.cuda()
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
    
    st = time.time()
    steps = 0
    cnn.train()
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        numpy.random.seed(seed=1)
        permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
        Xiter = X[permutation]
        Yiter = Y[permutation]

        while True:
             if i+batch_size < X_len:
                 batchX = Xiter[i:i+batch_size]
                 batchY = Yiter[i:i+batch_size]   
             else:
                 batchX = Xiter[i:]
                 batchY = Yiter[i:] 
#                 print('-------------%d----------------------'%i)

             character_ids = batchX.to(cuda)
             character_ids.contiguous()
#                 print('type',type(character_ids))
#                 print('size',character_ids.size())
             Xtensor = elmo(character_ids)
             Xtensor = Xtensor['elmo_representations'][0].float()
             Ytensor = torch.from_numpy(batchY).long()
             del batchX
             del batchY
             Xtensor = Xtensor.cuda()
             Ytensor = Ytensor.cuda()
             feature = Variable(Xtensor)
             target = Variable(Ytensor)
             del Xtensor
             del Ytensor
             i = i+batch_size

             optimizer.zero_grad() 
             logit = cnn(feature)
#             print(logit.size())    #
             loss = F.cross_entropy(logit, torch.max(target,1)[1])
             loss.backward()
             optimizer.step()

             steps += 1
             if i >= X_len:
                 break
        ct = time.time() - st
        unit = "s"
        if ct > 60:
            ct = ct/60
            unit = "m"
        print("time so far: ", str(ct), unit)
    return cnn
        
        
#        torch.save(model, modelfile)
        

#def elmo_cnn_model(X, Y, elmo,batch_size=16, learning_rate=0.001,num_epochs=10, loss_func='categorical_crossentropy'):
#    # Train the CNN, return the model
#    st = time.time()
#    Y = Y.astype('int') 
#    print("X numpy shape: ", str(X.size()), "Y numpy shape:", str(Y.shape))
#
#    # Params
#    X_len = X.size(0)
#    dim = X.size(-1)
#    num_labels = Y.shape[-1]
#    num_epochs = num_epochs
#    steps = 0
#    learning_rate = 0.001
#    cnn = CNN_Text(dim, num_labels)
#    cnn = cnn.cuda()
#
#    # Train
#    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)
#
#    steps = 0
#    cnn.train()
#    for epoch in range(num_epochs):
#        print("epoch", str(epoch))
#        i = 0
#        numpy.random.seed(seed=1)
#        permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
#        Xiter = X[permutation]
#        Yiter = Y[permutation]
#
#        while True:
#             if i+batch_size < X_len:
#    #             print(Xiter[i:i+batch_size].shape)
#                 batchX = Xiter[i:i+batch_size]
#                 batchY = Yiter[i:i+batch_size]
#             else:
#                 batchX = Xiter[i:]
#                 batchY = Yiter[i:]                
#             character_ids = batchX.to(cuda)
#             character_ids.contiguous()
##             print('type',type(character_ids))
##             print('size',character_ids.size())
#             Xtensor = elmo(character_ids)
#             Xtensor = Xtensor['elmo_representations'][0].float()
#             Ytensor = torch.from_numpy(batchY).long()
#             Xtensor = Xtensor.cuda()
#             Ytensor = Ytensor.cuda()
#             feature = Variable(Xtensor)
#             target = Variable(Ytensor)
#             i = i+batch_size
#
#             optimizer.zero_grad() 
#             logit = cnn(feature)
##             print(logit.size())    #
#             loss = F.cross_entropy(logit, torch.max(target,1)[1])
#             loss.backward()
#             optimizer.step()
#
#             steps += 1
#             del batchX
#             del batchY
#
#        # Print epoch time
#        ct = time.time() - st
#        unit = "s"
#        if ct > 60:
#            ct = ct/60
#            unit = "m"
#        print("time so far: ", str(ct), unit)
#    return cnn
def get_data(input_train):
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
def test(model,input_test,elmo,batch_size=16):
    print("testing...")
    stime = time.time()
    testlabels = []
    testX = []
    testY = []
    testX,testY,testlabels,testids = preprocess(input_test)
    
    y_pred = []
    logsoftmax = nn.LogSoftmax(dim=1)
    elmo = elmo.to(cuda)
    i = 0
#    print(testX.size(0),'size')
    while True:
        if i+batch_size<testX.size(0):
#        print(i)
            batchX = testX[i:i+batch_size]
        else:
            batchX = testX[i:]
        character_ids = batchX.to(cuda)
        character_ids.contiguous()
        Xtensor = elmo(character_ids)
        Xtensor = Xtensor['elmo_representations'][0].float()
        icd_var = model.forward(Variable(Xtensor))
        
        icd_vec = logsoftmax(icd_var)
        for j in range(icd_vec.size(0)):
#            print('icd_vec',icd_vec[i,:].size())
            icd_code = torch.max(icd_vec[j:j+1,:], 1)[1].data[0]
            icd_code = icd_code.item()
            y_pred.append(icd_code)
        i = i+batch_size
        if i >= testX.size(0):
            break
    print("testX shape: " + str(testX.shape))
    labenc = labelencoder
    predictedlabels = labenc.inverse_transform(y_pred)
    etime = time.time()
    print("testing took " + str(etime - stime) + " s")
    return testids, testlabels, predictedlabels
#options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
#weight_file = 'D:/projects/zhaodong/research/elmo_pubMed_only.hdf5'
#elmo = Elmo(options_file, weight_file, 1, dropout=0)
#sentences = ['First', 'sentence', '.']
#character_ids = batch_to_ids(sentences)
if __name__ == "__main__":main() 