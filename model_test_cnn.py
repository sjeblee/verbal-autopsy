#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu
'''
This file is for Serena's word embedding model, it's f1 score is really high, like 0.65 for child set
'''
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
#from layers import Attention

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

global anova_filter
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
    total_start_time = time.time()
    # Params
    num_feats = 500
    num_nodes = 192
    dropout = 0.5
    arg_rebalance = ""
    activation = "relu"
    arg_anova = "chi2"
    arg_train_feats = "D:/projects/zhaodong/research/va/data/dataset/train_child_cat_spell.features.narrv_08"
    arg_test_feats = "D:/projects/zhaodong/research/va/data/dataset/dev_child_cat_spell.features.narrv_08"
#    arg_train_feats = "/u/yanzhaod/data/va/mds+rct/train_child_cat_spell.features.narrv_08"
#    arg_test_feats = "/u/yanzhaod/data/va/mds+rct/dev_child_cat_spell.features.narrv_08"
    arg_prefix = "dev"
    global labelname
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
    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder()                        
    typeencoder = preprocessing.LabelEncoder()
    modelfile = "child_model.pt"
    stime = time.time()
    anova_filter = None

    print("anova_function: " + arg_anova)
    global model
    model = None
    cnn_model = None
    if train_mode== True:
        # Load the features
        print("Keys tested")
        Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, keys, trainlabels=True, Y2_labels='keyword_clusters')
        print("X: " + str(len(X)) + " Y: " + str(len(Y)))
        print("X2: " + str(len(X2)))
#        x_feats = numpy.asarray(X).shape[-1]
    
        # Rebalance
        if arg_rebalance != "":
            print("rebalance: " + arg_rebalance)
            X, Y = rebalance.rebalance(X, Y, arg_rebalance)
            print("X: " + str(len(X)) + "\nY: " + str(len(Y)))
    
        # Train the model
        print("training model...")
        print("creating a new neural network model")
        Y = to_categorical(Y)
        model = model_lib_test.cnn_model(X,Y)
        torch.save(model, modelfile)
    else:
        model = torch.load(modelfile)
    model.cuda()
    etime = time.time()
    print("training took " + str(etime - stime) + " s")
    testids, testlabels, predictedlabels = test(arg_model, model, arg_test_feats, anova_filter, hybrid, kw_cnn=cnn_model)
    print("Real Labels shape: " + str(testlabels))
    print("Predicted Labels shape: " + str(predictedlabels))
    f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
    print(f1score)
def test_cnn(model, testX, testids, probfile='/u/yoona/data/torch/probs_win200_epo10', labelencoder=None, collapse=False, threshold=0.1):
    y_pred = [] # Original prediction if threshold is not in used for ill-defined.
#    y_pred_softmax = []
#    y_pred_logsoftmax = []
#    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

#    new_y_pred = [] # class prediction if threshold for ill-difined is used.
    for x in range(len(testX)):
        input_row = testX[x]
   
        icd = None
        if icd is None:
            input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
            input_tensor = input_tensor.contiguous().cuda()
            icd_var = model(Variable(input_tensor))
            # Softmax and log softmax values
            icd_vec = logsoftmax(icd_var)
#            icd_vec_softmax = softmax(icd_var)
            icd_code = torch.max(icd_vec, 1)[1].data[0]
        icd_code = icd_code.item()
        y_pred.append(icd_code)
    return y_pred  # Comment this line out if threshold is not in used. 
def test(model_type, model, testfile, anova_filter=None, hybrid=False, rec_type=None, kw_cnn=None, threshold=0.01):
    print("testing...")
    stime = time.time()
    testids = []
    testlabels = []
    testX = []
    testX2 = []
    testY = []
    predictedY = []
    is_nn = ('nn' in model_type or model_type == "lstm" or model_type == "gru")
    if hybrid:
        vec_keys, point_keys = split_feats(keys, labelname)
        testY = preprocess(testfile, testids, testlabels, testX, testY, vec_keys)
        preprocess(testfile, [], [], testX2, [], point_keys)
    else:
        testY = preprocess(testfile, testids, testlabels, testX, testY, keys, rec_type, Y2_labels='keyword_clusters')
    if anova_filter is not None:
        testX = anova_filter.transform(testX)
    if is_nn:
        testX = numpy.asarray(testX)

    inputs = [testX]
    results = test_cnn(model,testX, testY,threshold=threshold)
    print("testX shape: " + str(testX.shape))
    #elif model_type == "cnn":
    #    attn_vec = get_attention_vector(model, testX)
    #    print "attention vector: " + str(attn_vec)  
#    global labelencoder
#    labelencoder = preprocessing.LabelEncoder()
    labenc = labelencoder
    if rec_type == 'adult':
        labenc = labelencoder_adult
    elif rec_type == 'child':
        labenc = labelencoder_child
    elif rec_type == 'neonate':
        labenc = labelencoder_neonate
    
    # Print out classes for index location of each class in the list
#    print("Index location of each class: ")
#    print(str(labenc.classes_))
    predictedlabels = labenc.inverse_transform(results)
    etime = time.time()
    print("testing took " + str(etime - stime) + " s")
#    print("testY: " + str(testY))
#    print("results: " + str(results))
#    print("predicted labels: " + str(predictedlabels))
    return testids, testlabels, predictedlabels
def preprocess(filename, ids, labels, x, y, feats, rec_type=None, trainlabels=False, Y2_labels=None):
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
    Y_arrays = []
    labels2 = []
    extra_labels = False
    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)
            features = []
            for key in keys:
            #for key in feats:
                if key == 'MG_ID':
                    ids.append(vector[key])
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

            x.append(features)
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
            x[i] = x[i] + cluster_feats[i]
        keys.remove("keyword_clusters")
        keys.append("keyword_clusters")
        #vec_feats = True
    labenc = labelencoder
    if trainlabels:
        labenc.fit(labels)
    else:
        labenc.fit(labels)
#    print("label",labels)
    y = labenc.transform(labels)
#    print("Y",y)
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

if __name__ == "__main__":main() 
