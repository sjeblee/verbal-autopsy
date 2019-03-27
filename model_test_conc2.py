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
    total_start_time = time.time()
    # Params
    num_feats = 500
    num_nodes = 192
    dropout = 0.5
    arg_rebalance = ""
    activation = "relu"
    arg_anova = "chi2"
    arg_train_feats = "D:/projects/zhaodong/research/va/data/dataset/train_adult_cat_spell.features.narrv_08"
    arg_test_feats = "D:/projects/zhaodong/research/va/data/dataset/dev_adult_cat_spell.features.narrv_08"
    #    arg_train_feats = "/u/yanzhaod/data/va/mds+rct/train_child_cat_spell.features.narrv_08"
    #    arg_test_feats = "/u/yanzhaod/data/va/mds+rct/dev_child_cat_spell.features.narrv_08"
    arg_prefix = "dev"
    global labelname
    global cuda
    cuda = torch.device("cuda:0")
    labelname = "cghr_cat"
    hybrid = False
    arg_model = "cnn"
    trainids = []        # VA record id
    trainlabels = []     # Correct ICD codes
    X = []               # Feature vectors
    Y = []
    X2 = [] # Extra features for hybrid model
    
    global keys
    with open(arg_train_feats + ".keys", "r") as kfile:
        keys = eval(kfile.read())
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models
    
    # If a mix of traditional and vector features is requested, use a hybrid nn model
    vec_keys, point_keys = split_feats(keys, labelname)
    
    # Transform ICD codes and record types to numbers
    global  typeencoder
    labelencoder = preprocessing.LabelEncoder()                        
    typeencoder = preprocessing.LabelEncoder()
    modelfile = "child_model.pt"
    stime = time.time()
    anova_filter = None
    input_train = "D:/projects/zhaodong/research/va/data/dataset/train_adult_cat_spell.xml"
    input_test = "D:/projects/zhaodong/research/va/data/dataset/dev_adult_cat.xml"
    print("anova_function: " + arg_anova)
    global emb_dic
    char_fname = 'char_emb/code/char_emb_30.txt'
    vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
    emb_dic = get_dic(char_fname,vocab)
    if train_mode== True:
        if to_preprocess == True:
            #-----------------collecting  gpu features char----------------------------
            char_dic,all_categories,max_num_char,n_iters = preprocess_char(input_train)
            #------------------------collecting cnn features---------------------------
            print("Keys tested")
            Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, keys, char_dic,trainlabels=True, Y2_labels='keyword_clusters')
            print("X: " + str(len(X)) + " Y: " + str(len(Y)))
            print("Y",Y)
            print("X2: " + str(len(X2)))
            # Train the model
            print("training model...")
            print("creating a new neural network model")
#            Y = to_categorical(Y)
            store_dic = {"all_categories":all_categories,
                         "max_num_char":max_num_char,
                         "n_iters":n_iters,}
            f = open("store_dic.txt","w")
            f.write(str(store_dic))
            f.close()
#            torch.save(X,"X.pt")
#            torch.save(Y,"Y.pt")
        else:
            f = open("store_dic.txt","r")
            store_dic= eval(f.read())
            all_categories = store_dic["all_categories"]
            max_num_char = store_dic["max_num_char"]
            n_iters = store_dic["n_iters"]
            X = torch.load('X.pt')
            Y = torch.load('Y.pt')
        model = combined_model(X,Y,all_categories,max_num_char,n_iters)
        torch.save(model, modelfile)

    else:
        f = open("store_dic.txt","r")
        store_dic= eval(f.read())
        all_categories = store_dic["all_categories"]
        max_num_char = store_dic["max_num_char"]
        model = torch.load(modelfile)
    etime = time.time()
    print("training took " + str(etime - stime) + " s")
    f1score = test(model,input_test,arg_test_feats,emb_dic,all_categories,max_num_char)
    print(f1score) 
#def test(model_type, model, testfile, anova_filter=None, hybrid=False, rec_type=None, kw_cnn=None, threshold=0.01):
#    print("testing...")
#    stime = time.time()
#    testids = []
#    testlabels = []
#    testX = []
#    testY = []
#    predictedY = []
#
#    testY = preprocess(testfile, testids, testlabels, testX, testY, keys, rec_type, Y2_labels='keyword_clusters')
#    testX = numpy.asarray(testX)
#
#    inputs = [testX]
#    results = test_cnn(model,testX, testY,threshold=threshold)
#    print("testX shape: " + str(testX.shape))
#    #elif model_type == "cnn":
#    #    attn_vec = get_attention_vector(model, testX)
#    #    print "attention vector: " + str(attn_vec)  
##    global labelencoder
##    labelencoder = preprocessing.LabelEncoder()
#    labenc = labelencoder
#    if rec_type == 'adult':
#        labenc = labelencoder_adult
#    elif rec_type == 'child':
#        labenc = labelencoder_child
#    elif rec_type == 'neonate':
#        labenc = labelencoder_neonate
#    
#    # Print out classes for index location of each class in the list
##    print("Index location of each class: ")
##    print(str(labenc.classes_))
#    predictedlabels = labenc.inverse_transform(results)
#    etime = time.time()
#    print("testing took " + str(etime - stime) + " s")
##    print("testY: " + str(testY))
##    print("results: " + str(results))
##    print("predicted labels: " + str(predictedlabels))
#    return testids, testlabels, predictedlabels

def preprocess(filename, ids, labels, x, y, feats,char_dic, rec_type=None, trainlabels=False, Y2_labels=None):
    global labelencoder, labelencoder_adult, labelencoder_child, labelencoder_neonate
    labelencoder  = preprocessing.LabelEncoder()
    #ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4","symp_vec"]
    ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4"]
    # Read in the feature vectors
    starttime = time.time()
    print("preprocessing features: " + str(feats))
    types = []
    kw_clusters = []
    vec_feats = False
    labels2 = []
    extra_labels = False
    mg = []
    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)      
            features = []
            for key in keys:
            #for key in feats:
                if key == 'MG_ID':
                    mg.append(vector[key])
                    ids.append(vector[key])
                    try:
                        char_feature = char_dic[vector[key]]
                    except KeyError:
                        print(vector[key])
                        char_feature = None
                elif key == labelname:
                    labels.append(vector[key])
                elif key in feats and key not in ignore_feats: # Only pull out the desired features
                    if key == "CL_type":
                        print("CL_type: " + vector[key])
                        types.append(vector[key])
                    elif key == "keyword_clusters":
                        kw_text = vector[key]
                        kw_list = []
                        if kw_text is not None:
                            kw_list = vector[key].split(',')
                        kw_clusters.append(kw_list)
                    elif key in vec_types:
                        # The feature matrix for word2vec can't have other features
                        #features = vector[key]
                        feature = vector[key]
                        vec_feats = True
                        if key == "narr_seq":
                            global vocab_size
                            vocab_size = vector['vocab_size']
                        global max_seq_len
                        max_seq_len = vector['max_seq_len']
                        #print "max_seq_len: " + str(max_seq_len)
                        #features = numpy.asarray(features)#.reshape(max_seq_len, 1)
                        feature = numpy.asarray(feature)
                        if len(features) == 0:
                            features = feature
                        else:
                            features = numpy.concatenate((features, feature), axis = 0)
            #print "narr_vec shape: " + str(features.shape)
                    elif not vec_feats:
                        if vector.has_key(key):
                    
                            features.append(vector[key])
                        else:
                            features.append('0')
            
            if char_feature is not None:
                x.append([torch.tensor(features,dtype=torch.float,device=cuda),char_feature])
    # Convert type features to numerical features
    if len(types) > 0: #and not vec_feats:
        print("converting types to numeric features")
        if trainlabels:
            typeencoder.fit(types)
        enc_types = typeencoder.transform(types)

        # Add the types back to the feature vector
        for i in range(len(x)):
            val = enc_types[i]
            x[i].append(val)
        keys.remove("CL_type")
        keys.append("CL_type")

    # Convert the keyword clusters to multi-hot vectors
    if len(kw_clusters) > 0:
        use_multi_hot = True
        #use_multi_hot = (Y2_labels == 'keyword_clusters')
        if use_multi_hot:
            print("converting keyword clusters to multi-hot vectors")
            cluster_feats = data_util.multi_hot_encoding(kw_clusters, 100)
            labels2 = cluster_feats
        else:
            # Convert keywords to cluster embeddings
            print("converting clusters to embeddings...")
            clusterfile = "/u/sjeblee/research/va/data/datasets/mds+rct/train_adult_cat_spell.clusters_e2"
            vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
            cluster_feats = cluster_keywords.cluster_embeddings(kw_clusters, clusterfile, vecfile)
            vec_feats = True

        # Update keys and features
        for i in range(len(x)):
            x[i][0] = x[i][0] + cluster_feats[i]
        keys.remove("keyword_clusters")
        keys.append("keyword_clusters")
        #vec_feats = True
    print('label',list(set(labels)))
    labenc = labelencoder
    labenc.fit(labels)
    y = labenc.transform(labels)
#    print("Y",y)
    #label is a list of string representation of categories,y is a list of encoded integers
    # Normalize features to 0 to 1 (if not word vectors)
    if not vec_feats:
    	preprocessing.minmax_scale(x, copy=False)
    endtime = time.time()
    mins = float(endtime-starttime)/60
    print("preprocessing took " + str(mins) + " mins")
    if extra_labels:
        print("returning extra labels")
        return [y, labels2]
    else:
        return y

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
    tensor = torch.zeros([max_num_char,emb_dim_char],device=cuda)
    for li, letter in enumerate(narrative):
        try:
            tensor[li] = torch.tensor(emb_dic[letter],device=cuda)
        except IndexError: #for test set, it's length may exceed max_num_char
            break
    return tensor

def getTensors(category,line,all_categories,max_num_char,emb_dim_char,emb_dic):
    category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long,device=cuda)
    #category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line,max_num_char,emb_dim_char,emb_dic)
    category_tensor = category_tensor.to(cuda)
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
def preprocess_char(input_train,n_hidden=128,emb_dim_char=30,learning_rate=0.0001,vocab='abcdefghijklmnopqrstuvwxyz0123456789 '):

    data,all_categories = get_data(input_train)
    print("vocab: %s" %vocab)
    n_iters = 0
    for k,v in data.items():
        n_iters += len(v)
    print("size of the narratives: %d" %n_iters)

    dic = {}
    max_num_char = 0
    for k,v in data.items():
        for i in range(len(v)):
            if len(v[i][1]) > max_num_char:
                max_num_char = len(v[i][1])
    for k,v in data.items():
        for i in range(len(v)):   
            category, line, category_tensor, line_tensor_char = getTensors(k,v[i][1],all_categories,max_num_char,emb_dim_char,emb_dic)
            dic[v[i][0]] = line_tensor_char
    print("max number of char %d" %max_num_char)

    return dic,all_categories,max_num_char,n_iters
def categoryFromOutput(output,all_categories):
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item()
    return all_categories[category_i], category_i

    
def combined_model(X, Y, all_categories,max_num_char,n_iters,n_hidden=100,emb_dim_char=30,emb_dim_word=100,act=None, windows=[1,2,3,4,5], X2=[], num_epochs=10, loss_func='categorical_crossentropy',dropout=0.0, kernel_sizes=5):
    # Train the CNN, return the model
    stime = time.time()
#    Yarray = Y.astype('int') 
    learning_rate = 0.0001

    batch_size = 16
    learning_rate = 0.001
    model = model_lib_test.CNN_GRU_Text(emb_dim_word, emb_dim_char, len(all_categories), n_hidden, max_num_char)
    model.cuda()
    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    training_set = []
    for i in range(len(X)):
        line_tensor_word,line_tensor_char = X[i][0],X[i][1]
        line_tensor_word = line_tensor_word.unsqueeze(1)
        category = Y[i]
        category_tensor = torch.tensor(category, dtype=torch.long,device=cuda)
        training_set.append([line_tensor_word,line_tensor_char,category_tensor,category])
    
    training_params = {"batch_size": batch_size,
                "shuffle": True,
                "num_workers": 0}
    training_generator = DataLoader(training_set, **training_params)    
    # Train
    print_every = 50
    print("Training...")
    count = 0
    epochs = 15
    all_losses = []
    current_loss = 0
    for i in range(epochs):
        for iter, batch in enumerate(training_generator):
            count += 1
            line_tensor_word, line_tensor_char, category_tensor,category= batch
            #train
            optimizer.zero_grad()  
            output,hidden = model.forward(line_tensor_word,line_tensor_char,None)
            #print(output.size(),category_tensor.size(),2)  ((batch,9);(batch,))
            loss = criterion(output, category_tensor)
            loss.backward()
            current_loss += loss
            optimizer.step()
            if count % print_every == 0:
                print('%d %d%% (%s) %.6f' % (count*batch_size, count*batch_size / n_iters/epochs*100, timeSince(stime),current_loss / print_every /batch_size)) 
                current_loss = 0
    print("Training took %s"%timeSince(stime))
    save(model,"model_test.pt")
    return model
def test(model,input_test,arg_train_test,emb_dic,all_categories,max_num_char,emb_dim_char=30):
    tdata,jumbo = get_data(input_test)
    length = 0
    for k,v in tdata.items():
        length += len(v)
    print("number of narratives in test set: %d" %length)
    dic = {}
    for k,v in tdata.items():
        for i in range(len(v)):
            category, line, category_tensor, line_tensor_char = getTensors(k,v[i][1],all_categories,max_num_char,emb_dim_char,emb_dic)
            dic[v[i][0]] = line_tensor_char
    X,Y = [],[]               # Feature vectors
    trainids,trainlabels = [],[]
    Y = preprocess(arg_train_test, trainids, trainlabels, X, Y, keys, dic,trainlabels=False, Y2_labels='keyword_clusters')
    print("X: " + str(len(X)) + " Y: " + str(len(Y)))
    batch_size = 8
    testing_set = []
    for i in range(len(X)):
        line_tensor_word,line_tensor_char = X[i][0],X[i][1]
        line_tensor_word = line_tensor_word.unsqueeze(1)
        category = Y[i]
        category_tensor = torch.tensor(category, dtype=torch.long,device=cuda)
        testing_set.append([line_tensor_word,line_tensor_char,category])
    testing_params = {"batch_size": batch_size,
                "shuffle": True,
                "num_workers": 0}
    testing_generator = DataLoader(testing_set, **testing_params)
    cat_pred,cat_true = [],[]
    for iter, batch in enumerate(testing_generator):
        line_tensor_word, line_tensor_char,category= batch 
        output,hidden = model.forward(line_tensor_word,line_tensor_char,None)
        l = []
        for i in range(output.size(0)):
            guess,guess_i = categoryFromOutput(output[i],all_categories)
            l.append(guess_i)
#        guess, guess_i = categoryFromOutput(output)
        cat_pred += l
        cat_true += [e for e in category]
    labenc = labelencoder
    cat_true = labenc.inverse_transform(cat_true)
    cat_pred = labenc.inverse_transform(cat_pred)
    print("Real Labels shape: " + str(cat_true))
    print("Predicted Labels shape: " + str(cat_pred))
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    return f1score
    
#def train_char(n_categories,n_hidden=100,emb_dim_char=30):
#    model = model_lib_test.GRU_Text(emb_dim_char, n_categories, n_hidden)
#    model.to(cuda)
    

if __name__ == "__main__":main() 
