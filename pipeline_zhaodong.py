import model_format
import sys
import numpy as np
import time
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn import preprocessing
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from lxml import etree
global anova_filter
global labelencoder
from allennlp.modules.elmo import Elmo, batch_to_ids
np.set_printoptions(threshold=np.inf)
import utils
from lxml import etree
from word2vec import load
from word2vec import get

import csv

#global all_categories
def main():
    global cuda
    global labelname
    global TRAINING, USE_SERVER, DEBUG, PRE_DATA, CROSS_VAL
    global max_num_word,max_char_in_word,emb_dim_char,emb_dim_comb,max_num_char
    global embedding, model_type, rec_type, dataset
    labelencoder = preprocessing.LabelEncoder()
    cuda = torch.device("cuda:0")
    #-----------------------mode----------------------------
    TRAINING = True        # To train the model; otherwise load existing model
    USE_SERVER = False      # To run in CSLab server, otherwise home computer
    DEBUG = True            # Debug mode
    PRE_DATA = True        # To preprocess the input data, otherwise load existing numpy array 
    CROSS_VAL = False        # To do cross validation on the ten-fold split sets (need to train and load data each time)
    embedding = 'conc'       # Choices: 'conc', 'elmo'
    model_type = 'cnn' 
    rec_type = "child"      # Choices: 'child','neonate','adult'
    dataset = 'mds'       # Choices: 'mds','phmrc'
    #---------------------parameters-----------------------
    learning_rate = 0.003
    num_epochs = 12
    batch_size= 16
    max_num_word = 200
    emb_dim_char= 24
    max_char_in_word = 7
    max_num_char = 1000
    #---------------initialized variables----------------
    emb_dim_word = 100
    emb_dim_comb = max_char_in_word*emb_dim_char+emb_dim_word
    total_start_time = time.time()
    prefix = '-'.join([embedding,model_type+rec_type])
    if dataset == 'phmrc':
        prefix += '_phmrc'
    out_dir = 'output/'
    data_file_X = out_dir+prefix+'_train.npy'
    data_file_testX = out_dir+prefix+'_test.npy'
    data_labels = out_dir+prefix+'_labels.npy'
    data_testlabels = out_dir+prefix+'_testlabels.npy'
    data_testIDs = out_dir+prefix+'_testids.npy'
    data_ids = out_dir+prefix+'_ids.npy'
    modelfile = prefix+'_model.pt'    # Filename of the saved model
    #---------------------------------
    if USE_SERVER:
        if dataset == 'mds':
            input_train = "/u/yanzhaod/data/va/mds+rct/train"+rec_type+"_cat_spell.xml"  #input train file for char_embeeding
            input_test = "/u/yanzhaod/data/va/mds+rct/test"+rec_type+"_cat_spell.xml"    #input test file for char_embedding

    else:
        if dataset == 'mds':
            input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+rec_type+"_cat_spell.xml"
            input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+rec_type+"_cat_spell.xml"
        elif dataset == 'phmrc':
            input_train = "D:/projects/zhaodong/research/va/data/phmrc_data/train_adult_cat.xml"  #input train file for char_embeeding
            input_test = "D:/projects/zhaodong/research/va/data/phmrc_data/test_adult_cat.xml"    #input test file for char_embedding
    if CROSS_VAL:
        for i in range(10):
            input_train = "D:/projects/zhaodong/research/va/data/crossval_sets/train_"+rec_type+"_"+str(i)+"_cat_spell.xml"  #input train file for char_embeeding
            input_test = "D:/projects/zhaodong/research/va/data/crossval_sets/test_"+rec_type+"_"+str(i)+"_cat_spell.xml"    #input test file for char_embedding
            X,labels,IDs = preprocess(input_train,i)
            testX,testlabels,testIDs = preprocess(input_test,i)
            all_labels = np.concatenate([labels,testlabels])
            labenc = labelencoder
            labenc.fit(all_labels)
            Y = labenc.transform(labels)
            Y = to_categorical(Y)
            all_categories = list(set(list(all_labels)))
            print("all categories: "+str(all_categories))
            class_num = len(all_categories)
            if embedding == 'conc':
                model = model_format.CNN_TEXT(emb_dim_comb,class_num,dropout=0.0, kernel_sizes=5)
                model.fit(X,Y,emb_dim_comb,learning_rate=learning_rate,batch_size=batch_size,num_epochs=num_epochs)
            elif embedding == 'elmo':
                emb_dim = 1024
                model = model_format.CNN_ELMO(emb_dim,class_num,USE_SERVER=USE_SERVER)
                model.fit(X,Y,num_epochs=num_epochs,batch_size=batch_size,learning_rate=learning_rate)
            y_pred = model.predict(testX)
            labenc = labelencoder
            predictedlabels = labenc.inverse_transform(y_pred)
            f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
            precision = metrics.precision_score(testlabels, predictedlabels, average='weighted')
            recall = metrics.recall_score(testlabels, predictedlabels, average='weighted')
    else:
        if PRE_DATA:
            X,labels,IDs = preprocess(input_train)
            testX,testlabels,testIDs = preprocess(input_test)
            utils.save_numpy([X,labels,IDs,testX,testlabels,testIDs],
                             [data_file_X,data_labels,data_ids,data_file_testX,data_testlabels,data_testIDs])
        else:
            (X,labels,IDs,testX,testlabels,testIDs) = utils.load_numpy([data_file_X,data_labels,data_ids,
                                                                        data_file_testX,data_testlabels,data_testIDs])
        all_labels = np.concatenate([labels,testlabels])
        labenc = labelencoder
        labenc.fit(all_labels)
        Y = labenc.transform(labels)
        Y = to_categorical(Y)
        all_categories = list(set(list(all_labels)))
        print("all categories: "+str(all_categories))
        class_num = len(all_categories)
        
        if TRAINING:
            if embedding == 'conc':
                model = model_format.CNN_TEXT(emb_dim_comb,class_num,dropout=0.0, kernel_sizes=5)
                model.fit(X,Y,emb_dim_comb,learning_rate=learning_rate,batch_size=batch_size,num_epochs=num_epochs)
            elif embedding == 'elmo':
                emb_dim = 1024
                model = model_format.CNN_ELMO(emb_dim,class_num,USE_SERVER=USE_SERVER)
                model.fit(X,Y,num_epochs=num_epochs,batch_size=batch_size,learning_rate=learning_rate)
            elif embedding == 'char':
                model = model_format.CNN_TEXT(emb_dim_char,class_num,dropout=0.0, kernel_sizes=5)
                model.fit(X,Y,emb_dim_char,learning_rate=learning_rate,batch_size=batch_size,num_epochs=num_epochs)
            torch.save(model, modelfile)
        else:
            model = torch.load(modelfile)
        y_pred = model.predict(testX)
        labenc = labelencoder
        predictedlabels = labenc.inverse_transform(y_pred)
        print('--------------Final results--------------------')
        precision, recall, f1score, csmf_accuracy = utils.stats_from_results(testlabels,predictedlabels,testIDs,PRINT=True)
        print("Overall it takes " + utils.timeSince(total_start_time))

def preprocess(input_train,ind=0):
    '''
    INPUT:
        input_train: path to the training xml file
    OUTPUT:
        X: a list of strings, containing the narratives
        ID: a list of strings, containing MG_IDs
        label: numpy array of string, containing the labels
    '''
    if dataset == 'phmrc':
        def csv_to_dic(file):
            with open(file, mode='r') as infile:
                reader = csv.reader(infile)
                mydict = {rows[0]:rows[1] for rows in reader}
            return mydict
        label_dic = csv_to_dic("D:/projects/zhaodong/research/va/data/phmrc_data/map_phmrc_to_cghr_adult.csv")
        print('label dictionary: '+str(label_dic))
        data,all_categories = utils.get_data_phmrc(input_train,label_dic)
    else:
        data,all_categories = utils.get_data(input_train)
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
        label = np.array(label)
        X = batch_to_ids(sentence)   
        del sentence
        if X.size(1) > max_num_word:
            X = X[:,:max_num_word,:]
        else:
            p2d = (0, 0, 0, max_num_word-X.size(1)) 
            X = F.pad(X, p2d, 'constant', 0)
        if DEBUG:
            print(type(X),'11111111111111')
    elif embedding == 'conc':
        char_fname = 'char_emb/code/char_emb_'+str(emb_dim_char)+'.txt'
        vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
        emb_dic = utils.get_dic(char_fname,vocab)
        if CROSS_VAL:
            wmodel,dim = load('D:/projects/zhaodong/research/va/data/crossval_sets/ice+medhelp+narr_all_'+str(ind)+'.vectors.100')
        else:
            if USE_SERVER:
                wmodel,dim = load('/u/yanzhaod/data/narr+ice+medhelp.vectors.100')
            else:
                wmodel,dim = load('D:/projects/zhaodong/research/va/data/dataset/narr+ice+medhelp.vectors.100')
        def letterToNumpy(letter):
            return np.array(emb_dic[letter])
        def lettersToNumpy(word):
            arr = np.zeros((max_char_in_word,emb_dim_char))
            for i in range(max_char_in_word):
                if i < len(word):
                    arr[i,:] = letterToNumpy(word[i])
            return arr
        def wordToNumpy(word):
            emb_word = np.array(get(word,wmodel))
            emb_letters = lettersToNumpy(word).flatten('C')    #'C' means to flatten in row major  #of shaoe(max_char_in_word*emb_dim_char,)
            return np.hstack((emb_word,emb_letters))    #(max_char_in_word*emb_dim_char+emb_dim_word,)
        def lineToNumpy(line):
            '''
            INPUT:
                line: string
            OUTPUT:
                emb_lin: numpy array of size (max_num_word,emb_dim_comb)
            '''
            l = line.split()
            emb_line = np.zeros((max_num_word,emb_dim_comb))
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
    elif embedding == 'char':
        char_fname = 'char_emb/code/char_emb_'+str(emb_dim_char)+'.txt'
        vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
        emb_dic = utils.get_dic(char_fname,vocab)
        def letterToNumpy(letter):
            return numpy.array(emb_dic[letter])
        def lineToNumpy(line):
            emb_line = np.zeros((max_num_char,emb_dim_char))
            for i in range(max_num_char):
                if i < len(line):
                    emb_line[i,:] = letterToNumpy(line[i])
            return emb_line
        data,all_categories = utils.get_data(input_train)
        ID = []
        X,label = [],[]
        for k,v in data.items():
            for i in range(len(v)):
                ID.append(v[i][0])
                X.append(lineToNumpy(v[i][1]))
                label.append(k)
    X = np.asarray(X).astype('float')
    return X,label,ID

if __name__ == "__main__":main() 
