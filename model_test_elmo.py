#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import sys

import numpy

import time

from keras.utils.np_utils import to_categorical
from sklearn import metrics

from sklearn import preprocessing
import statistics
#import attention_utils
from model_lib_test import CNN_Text
import math
#from layers import Attention
import re
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

# Output top K features
output_topk_features = True

def main():
    global labelname
    global cuda
    global keys
    global emb_dic
    global  typeencoder
    
    #--------------------parameters----------------------
    learning_rate = 0.003
    num_epochs = 12
    batch_size= 16
    emb_dim_char = 50
    rec_type = "adult"
    train_mode = True     #True to start a new training, False for load existing model
    total_start_time = time.time()
    #-----------------initialize variables-----------
    cuda = torch.device("cuda:0")
    labelname = "cghr_cat"                     
    typeencoder = preprocessing.LabelEncoder()
    modelfile = rec_type+"_model.pt"     #the model file to be saved or loaded

    #---------------------elmo module-----------------------
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/contributed/pubmed/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = '/u/yanzhaod/data/elmo_pubMed_only.hdf5'
    global elmo
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
            X,Y,label,ID= preprocess(input_train)
    #        print("X: " + str(len(X)) +" Y: " + str(len(Y)))
            Y = to_categorical(Y)
            model = elmo_cnn_model(X,Y,elmo,cuda,learning_rate=learning_rate,batch_size=batch_size,num_epochs=num_epochs)
            torch.save(model, modelfile)
        else:
            model = torch.load(modelfile)
        testids, testlabels, predictedlabels = test(model,input_test,emb_dim_char=emb_dim_char)
        print("Real Labels shape: " + str(testlabels))
        print("Predicted Labels shape: " + str(predictedlabels))
        f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
        precision = metrics.precision_score(testlabels, predictedlabels, average='weighted')
        recall = metrics.recall_score(testlabels, predictedlabels, average='weighted')
        result  =[]
        for j in range(len(testids)):
            result.append({'Correct_ICD':testlabels[j],'Predicted_ICD':predictedlabels[j],'MG_ID':testids[j]})
        csmf_accuracy = stats_from_result(result)
        print('precision: '+str(precision))
        print('recall: '+str(recall))
        print('f1score: '+str(f1score))
        print('csmf_accuracy: '+str(csmf_accuracy))
        f1scores.append(f1score)
        precisions.append(precision)
        recalls.append(recall)
        csmf.append(csmf_accuracy)
    print('--------------Final results--------------------')
    print("Precision: "+str(statistics.mean(precisions)))
    print("Recall: "+str(statistics.mean(recalls)))    
    print("F1score: "+str(statistics.mean(f1scores)))
    print("Csmf accuracy: "+str(statistics.mean(csmf)))
    print("Overall it takes " + timeSince(total_start_time))

def stats_from_result(result):
    '''
    
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
def test(model,input_test,emb_dim_char=24, hybrid=False, rec_type=None, kw_cnn=None, threshold=0.01):
    print("testing...")
    stime = time.time()
    testlabels = []
    testX = []
    testY = []
    testX,testY,testlabels,emb_dim_comb,testids = preprocess(input_test,emb_dim_char=emb_dim_char)

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
def preprocess(input_file,n_hidden=128,emb_dim=50,max_num_word = 200,max_char_in_word = 7,learning_rate=0.0001,vocab='abcdefghijklmnopqrstuvwxyz0123456789 '):
    global labelencoder
    labelencoder  = preprocessing.LabelEncoder()
    data,all_categories = get_data(input_file)
    print("vocab: %s" %vocab)
    n_iters = 0
    for k,v in data.items():
        n_iters += len(v)
    print("size of the narratives: %d" %n_iters)

    global sentence
    global character_ids
    ID = []
    label = []
    sentence = []
    for k,v in data.items():
        for i in range(len(v)):
            ID.append(v[i][0])
            label.append(k)
            text = v[i][1].split()
            sentence.append(text)
    
    X = batch_to_ids(sentence)
    X = X.to(cuda)
    X.contiguous()
#    X = X.to(cuda)
#    print("matrix size",X.size())   #num_sentence*max_char_in_word*embedding_size
    
    labenc = labelencoder
    label = numpy.array(label)
#    real_label = ['1', '10', '11', '12', '13', '14', '15', '16', '17', '18', '2', '3', '4', '5', '6', '7', '8', '9']
    labenc.fit(label)
    Y = labenc.transform(label)
    del data
    return X,Y,label,ID
def elmo_cnn_model(X, Y, elmo,batch_size=16, learning_rate=0.001,num_epochs=10, loss_func='categorical_crossentropy'):
    # Train the CNN, return the model
    st = time.time()
    Y = Y.astype('int') 
    print("X numpy shape: ", str(X.size()), "Y numpy shape:", str(Y.shape))

    # Params
    X_len = X.size(0)
    dim = X.size(-1)
    num_labels = Y.shape[-1]
    num_epochs = num_epochs
    steps = 0
    learning_rate = 0.001
    cnn = CNN_Text(dim, num_labels,dropout=dropout, kernel_sizes=kernel_sizes)
    cnn = cnn.cuda()

    # Train
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    steps = 0
    cnn.train()
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        numpy.random.seed(seed=1)
        permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
        Xiter = X[permutation]
        Yiter = Y[permutation]

        while i+batch_size < X_len:
             print(Xiter[i:i+batch_size].shape)
             batchX = Xiter[i:i+batch_size]
             batchY = Yiter[i:i+batch_size]
             character_ids = batchX.to(cuda)
             character_ids.contiguous()
             print('type',type(character_ids))
             print('size',character_ids.size())
             Xtensor = elmo(character_ids)['elmo_representations'][0].float()
             Ytensor = torch.from_numpy(batchY).long()
             Xtensor = Xtensor.cuda()
             Ytensor = Ytensor.cuda()
             feature = Variable(Xtensor)
             target = Variable(Ytensor)
             i = i+batch_size

             optimizer.zero_grad() 
             logit = cnn(feature)
#             print(logit.size())    #
             loss = F.cross_entropy(logit, torch.max(target,1)[1])
             loss.backward()
             optimizer.step()

             steps += 1
             del batchX
             del batchY
        # Print epoch time
        ct = time.time() - st
        unit = "s"
        if ct > 60:
            ct = ct/60
            unit = "m"
        print("time so far: ", str(ct), unit)
    return cnn
if __name__ == "__main__":main() 
