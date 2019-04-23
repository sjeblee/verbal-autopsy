#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import sys
sys.path.append('../keras-attention-mechanism')
sys.path.append('keywords')

import argparse
import numpy

import time

from keras.utils.np_utils import to_categorical
from sklearn import metrics

from sklearn import preprocessing

from sklearn.feature_selection import SelectKBest, f_classif, chi2

#import attention_utils
import cluster_keywords
import data_util
import rebalance
import model_lib_test
import math
#from layers import Attention
import re
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from lxml import etree
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
import math
global anova_filter
global labelencoder
import ast
from word2vec import load
from word2vec import get
labelencoder = None
labelencoder_adult = None
labelencoder_child = None
labelencoder_neonate = None
vec_types = ["narr_vec", "narr_seq", "event_vec", "event_seq", "symp_vec", "kw_vec", "textrank_vec"]
numpy.set_printoptions(threshold=numpy.inf)

# Output top K features
output_topk_features = True

'''
The difference compared with the 

'''
def main():
    train_mode = True
    to_preprocess = True
    learning_rate = 0.001
    num_epochs = 10
    batch_size= 16
    total_start_time = time.time()
    rec_type = "child"
    arg_train_feats = "D:/projects/zhaodong/research/va/data/dataset/train_"+rec_type+"_cat_spell.features.narrv_08"
    arg_test_feats = "D:/projects/zhaodong/research/va/data/dataset/dev_"+rec_type+"_cat_spell.features.narrv_08"
    #    arg_train_feats = "/u/yanzhaod/data/va/mds+rct/train_child_cat_spell.features.narrv_08"
    #    arg_test_feats = "/u/yanzhaod/data/va/mds+rct/dev_child_cat_spell.features.narrv_08"
    global labelname
    global cuda
    global keys
    global emb_dic
    global  typeencoder
    cuda = torch.device("cuda:0")
    labelname = "cghr_cat"
    trainids,trainlabels = [],[]        # VA record id # Correct ICD codes
    X,Y=[],[]   # Feature vectors
    X2 = [] # Extra features for hybrid model
    
    with open(arg_train_feats + ".keys", "r") as kfile:
        keys = eval(kfile.read())
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models
    vec_keys, point_keys = split_feats(keys, labelname)

    labelencoder = preprocessing.LabelEncoder()                        
    typeencoder = preprocessing.LabelEncoder()
    modelfile = rec_type+"_model.pt"
    stime = time.time()
    input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+rec_type+"_cat_spell.xml"  #input train file for char_embeeding
    input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+rec_type+"_cat_spell.xml"      #input test file for char_embedding
    char_fname = 'char_emb/code/char_emb_30.txt'         #pretrained char_embeddings
    vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
    emb_dic = get_dic(char_fname,vocab)
    if train_mode== True:
        X,Y,label,emb_comb,ID= preprocess(input_train)
        print("X: " + str(len(X)) + " X2: " + str(len(X2)) +" Y: " + str(len(Y)))
        Y = to_categorical(Y)
        model = model_lib_test.cnn_comb_model(X,Y,emb_comb,learning_rate=learning_rate,batch_size=batch_size,num_epochs=num_epochs)
        torch.save(model, modelfile)

    else:
        model = torch.load(modelfile)
    etime = time.time()
    print("training took " + str(etime - stime) + " s")
    testids, testlabels, predictedlabels = test(model,input_test,arg_test_feats)
    print("Real Labels shape: " + str(testlabels))
    print("Predicted Labels shape: " + str(predictedlabels))

    f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
    print(f1score)
    print("Overall it takes " + timeSince(total_start_time))
    result  =[]
    for i in range(len(testids)):
        result.append({'Correct_ICD':testlabels[i],'Predicted_ICD':predictedlabels[i],'MG_ID':testids[i]})
    for i in range(len(result)):
        result[i] = str(result[i])
    f = open('result_for_confusion_mat.txt','w')
    f.write('\n'.join(result))
    f.close()

def test(model,input_test, testfile,emb_dim_char=30,anova_filter=None, hybrid=False, rec_type=None, kw_cnn=None, threshold=0.01):
    print("testing...")
    stime = time.time()
    testlabels = []
    testX = []
    testY = []
    testX,testY,testlabels,emb_dim_comb,testids = preprocess(input_test)

    testX = numpy.asarray(testX).astype('float')
    results = test_both(model,testX, testY,threshold=threshold)
    print("testX shape: " + str(testX.shape))
    #elif model_type == "cnn":
    #    attn_vec = get_attention_vector(model, testX)
    #    print "attention vector: " + str(attn_vec)  
#    global labelencoder
#    labelencoder = preprocessing.LabelEncoder()
    labenc = labelencoder
    predictedlabels = labenc.inverse_transform(results)
    etime = time.time()
    print("testing took " + str(etime - stime) + " s")
#    print("testY: " + str(testY))
#    print("results: " + str(results))
#    print("predicted labels: " + str(predictedlabels))
    return testids, testlabels, predictedlabels
def test_both(model, testX, testids, probfile='/u/yoona/data/torch/probs_win200_epo10', labelencoder=None, collapse=False, threshold=0.1):
    y_pred = [] # Original prediction if threshold is not in used for ill-defined.
#    y_pred_softmax = []
#    y_pred_logsoftmax = []
#    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
#    testX2 = numpy.asarray(testX2).astype('float')
#    new_y_pred = [] # class prediction if threshold for ill-difined is used.
    for x in range(len(testX)):
        input_row = testX[x]
        icd = None
        if icd is None:
            input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
            input_tensor = input_tensor.contiguous().cuda()
            icd_var = model.forward(Variable(input_tensor))
            # Softmax and log softmax values
            icd_vec = logsoftmax(icd_var)
#            icd_vec_softmax = softmax(icd_var)
            icd_code = torch.max(icd_vec, 1)[1].data[0]
        icd_code = icd_code.item()
        y_pred.append(icd_code)
    return y_pred  # Comment this line out if threshold is not in used. 

def split_feats(keys, labelname):
    ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4"] 
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models
    for key in keys:
        if key in vec_types:
            vec_keys.append(key)
        elif key == labelname or key not in ignore_feats:
            point_keys.append(key)

    print("vec_keys: " + str(vec_keys))
    print("point_keys: " + str(point_keys))
    print("Keys printed")
    return vec_keys, point_keys

#########################################################
# Select K Best symptoms for each class
# Create output files containing best k features for each class
# Arguments
# 	X		: list of features
# 	Y		: ndarray after labelencoder transformation
# 	function	: type of anova function (ex. f_classif, chi2)
#	output_path	: path to the output file
# 	k		: number of top-k features to be selected
#
#
def select_top_k_features_per_class(X, Y, function, output_path, k = 100):
    classes = labelencoder.classes_

    for i in range(len(classes)):
        output = open(output_path + "/top_" + str(k) + "_features_class_" + classes[i], 'w')
        output.write("Class : " + str(classes[i]))
        print("Class: " + str(classes[i]))
        this_Y = []
        for j in range(len(Y)):
            if Y[j] == i:
                binary= 1
            else:
                binary= 0
            this_Y.append(binary)
        anova_symp = SelectKBest(function, 'all')
        anova_symp.fit(X,this_Y)
        best_indices = anova_symp.get_support(True)
        scores = anova_symp.scores_
        output.write("The sorted indices:")
        sorted_idx = numpy.argsort(scores)[::-1][:k]
        output.write(str(sorted_idx))
    
        for i in range(len(sorted_idx)):
            selected = str(keys[sorted_idx[i] + 2])
            put.write("\n")
            put.write(selected + " ")
            output.write(str(scores[sorted_idx[i]]))
        output.close()

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

def get_dic(fname,vocab):
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

def lineToTensor(narrative,max_num_char,emb_dim_char,emb_dic):
    tensor = torch.zeros([max_num_char,emb_dim_char])
    for li, letter in enumerate(narrative):
        try:
            tensor[li] = torch.tensor(emb_dic[letter])
        except IndexError: #for test set, it's length may exceed max_num_char
            break
    return tensor

def getTensors(category,line,all_categories,max_num_char,emb_dim_char,emb_dic):
    category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long)
    #category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line,max_num_char,emb_dim_char,emb_dic)
    return category, line, category_tensor, line_tensor
def save(model,out_model_filename):
    torch.save(model, out_model_filename)
    print('Saved as %s' % out_model_filename)
   
def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)
def preprocess(input_file,n_hidden=128,emb_dim_char=30,emb_dim_word = 100,max_num_word = 200,max_char_in_word = 7,learning_rate=0.0001,vocab='abcdefghijklmnopqrstuvwxyz0123456789 '):
    global labelencoder
    labelencoder  = preprocessing.LabelEncoder()
    wmodel,dim = load('D:/projects/zhaodong/research/va/data/dataset/narr+ice+medhelp.vectors.100')
    emb_dim_comb = max_char_in_word*emb_dim_char+emb_dim_word
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
        l = line.split()
        emb_line = numpy.zeros((max_num_word,emb_dim_comb))
        for i in range(max_num_word):
            if i < len(l):
                emb_line[i,:] = wordToNumpy(l[i])
        return emb_line
    data,all_categories = get_data(input_file)
    print("vocab: %s" %vocab)
    n_iters = 0
    for k,v in data.items():
        n_iters += len(v)
    print("size of the narratives: %d" %n_iters)
    
    ID = []
    X,label = [],[]
    for k,v in data.items():
        for i in range(len(v)):
            ID.append(v[i][0])
            X.append(lineToNumpy(v[i][1]))
            label.append(k)
    label = numpy.array(label)
    labenc = labelencoder
    labenc.fit(label)
    Y = labenc.transform(label)
    return X,Y,label,emb_dim_comb,ID

if __name__ == "__main__":main() 
