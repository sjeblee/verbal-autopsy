import model_format
#import sys
import numpy as np
import time
import os
from keras.utils.np_utils import to_categorical
#from sklearn import metrics
import sklearn
import torch
#import torch.nn as nn
#from torch.autograd import Variable
import torch.nn.functional as F
#from lxml import etree
global anova_filter
global labelencoder

np.set_printoptions(threshold=np.inf)
import utils
#from lxml import etree
from word2vec import load
from word2vec import get
import statistics
import csv
import parameters
from pytorch_pretrained_bert import BertTokenizer
from tqdm import tqdm
from collections import OrderedDict
from pytorch_pretrained_bert.modeling import BertConfig
from pytorch_pretrained_bert import BertModel
import preprocessing

def main():
#    global X  #to be deleted
    global cuda
    global labelname,feat
#    global TRAINING, USE_SERVER, DEBUG, PRE_DATA, CROSS_VAL
    global USE_SERVER
    global max_num_word,max_char_in_word,emb_dim_char,emb_dim_comb,max_num_char,emb_dim_word
    global embedding, model_type, rec_type, dataset
    labelencoder = sklearn.preprocessing.LabelEncoder()
    cuda = torch.device("cuda:0")
    if 'elmo' in parameters.embedding:
        from allennlp.modules.elmo import Elmo, batch_to_ids
#    #---------------------parameters-----------------------
#    
    max_num_word = parameters.max_num_word
    emb_dim_char= parameters.emb_dim_char
    max_char_in_word = parameters.max_char_in_word
    max_num_char = parameters.max_num_char
    feat = ['region','Language']    #can be either a feature string or a list of feature strings
                                    #current feature: 'region','Language','StateCode'
    #---------------initialized variables----------------
    if os.path.exists('D:/projects/Git/verbal-autopsy'):
        USE_SERVER = False
    else:
        USE_SERVER = True
    
    emb_dim_word = 100
    emb_dim_comb = max_char_in_word*emb_dim_char+emb_dim_word
    total_start_time = time.time()
    prefix = '_'.join([parameters.embedding,parameters.model_type,parameters.rec_type,parameters.dataset])
    out_dir = 'output/'
    data_file_X = out_dir+prefix+'_train.npy'
    data_file_testX = out_dir+prefix+'_test.npy'
    data_labels = out_dir+prefix+'_labels.npy'
    data_testlabels = out_dir+prefix+'_testlabels.npy'
    data_testIDs = out_dir+prefix+'_testids.npy'
    data_ids = out_dir+prefix+'_ids.npy'
    data_X2 = out_dir+prefix+'_X2.npy'
    data_testX2 = out_dir+prefix+'_testX2.npy'
    modelfile = out_dir+'model/'+prefix+'_model.pt'    # Filename of the saved model
    #---------------------------------
    if USE_SERVER:
        if parameters.dataset == 'mds':
            if parameters.rec_type != 'neonate':
                input_train = "/u/yanzhaod/data/va/mds+rct/train_"+parameters.rec_type+"_cat_spell.xml"  #input train file for char_embeeding
                input_test = "/u/yanzhaod/data/va/mds+rct/test_"+parameters.rec_type+"_cat_spell.xml"    #input test file for char_embedding
            else:
                input_train = "/u/yanzhaod/data/va/mds+rct/train_"+parameters.rec_type+"_cat.xml"  #input train file for char_embeeding
                input_test = "/u/yanzhaod/data/va/mds+rct/test_"+parameters.rec_type+"_cat.xml"    #input test file for char_embedding               

    else:
        if parameters.dataset == 'mds':
            if parameters.rec_type != 'neonate':
                input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+parameters.rec_type+"_cat_spell.xml"
                input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+parameters.rec_type+"_cat_spell.xml"
            else:
                input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+parameters.rec_type+"_cat_spell.xml"
                input_test = "D:/projects/zhaodong/research/va/data/dataset/dev_"+parameters.rec_type+"_cat_spell.xml"               
        elif parameters.dataset == 'phmrc':
            input_train = "D:/projects/zhaodong/research/va/data/phmrc_data/train_adult_cat.xml"  #input train file for char_embeeding
            input_test = "D:/projects/zhaodong/research/va/data/phmrc_data/test_adult_cat.xml"    #input test file for char_embedding
    if parameters.CROSS_VAL:
        f1scores,precisions,recalls,csmfs = [],[],[],[]
        for i in range(10):
            if USE_SERVER:
                input_train = "/u/yanzhaod/data/va/crossval_sets/train_"+parameters.rec_type+"_"+str(i)+"_cat_spell.xml"  #input train file for char_embeeding
                input_test = "/u/yanzhaod/data/va/crossval_sets/test_"+parameters.rec_type+"_"+str(i)+"_cat_spell.xml"    #input test file for char_embedding               
            else:
                input_train = "D:/projects/zhaodong/research/va/data/crossval_sets/train_"+parameters.rec_type+"_"+str(i)+"_cat_spell.xml"  #input train file for char_embeeding
                input_test = "D:/projects/zhaodong/research/va/data/crossval_sets/test_"+parameters.rec_type+"_"+str(i)+"_cat_spell.xml"    #input test file for char_embedding
            X,labels,IDs,X2 = preprocess(input_train)
            testX,testlabels,testIDs,testX2 = preprocess(input_test)
            if 'struct' in parameters.embedding:
                def repeat_arr(arr,rep):
                    '''
                    repeat arrs for larger dimension
                    '''
                    temp = arr
                    for i in range(rep-1):
                        arr = np.hstack((arr,temp))
                    return arr
                X2 = repeat_arr(X2,parameters.rep)
                testX2 = repeat_arr(testX2,parameters.rep)
                emb_dim_feat = X2.shape[1]
                print('embeded dimension of structured features: '+str(emb_dim_feat))
                print('X.shape: '+str(X.shape)+';  X2.shape: '+str(X2.shape))
                print('testX.shape: '+str(testX.shape)+';  testX2.shape: '+str(testX2.shape))
            all_labels = np.concatenate([labels,testlabels])
            labenc = labelencoder
            labenc.fit(all_labels)
            Y = labenc.transform(labels)
            
            Y = to_categorical(Y)
            all_categories = list(set(list(all_labels)))
            print("all categories: "+str(all_categories))
            class_num = len(all_categories)
            if parameters.embedding == 'conc':
                print('embedding: combination of word and character embedding at input level')
                model = model_format.CNN_TEXT(emb_dim_comb,class_num,dropout=0.0, kernel_sizes=5)
                model.fit(X,Y,emb_dim_comb,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'elmo':
                print('embedding: pubMed elmo embedding')
                emb_dim = 1024
                model = model_format.CNN_ELMO(emb_dim,class_num,USE_SERVER=USE_SERVER)
                model.fit(X,Y,num_epochs=parameters.num_epochs,batch_size=parameters.batch_size,
                          learning_rate=parameters.learning_rate,dropout=parameters.dropout)
            elif parameters.embedding == 'struct-elmo':
                emb_dim = 1024
                print('embedding: narrative data with structured features with elmo')
                model = model_format.STRUCT_ELMO(emb_dim,emb_dim_feat,class_num,dropout=0.0, kernel_sizes=5, USE_SERVER=USE_SERVER)
                model.fit(X,X2,Y,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'struct':
                print('embedding: narrative data with structured features')
                model = model_format.CNN_STRUCT(emb_dim_comb,emb_dim_feat,class_num,dropout=0.0, kernel_sizes=5)
                model.fit(X,X2,Y,emb_dim_comb,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            y_pred = model.predict(testX,testX2)
            labenc = labelencoder
            predictedlabels = labenc.inverse_transform(y_pred)
            precision, recall, f1score, csmf_accuracy = utils.stats_from_results(testlabels,predictedlabels,testIDs,PRINT=True)
            f1scores.append(f1score)
            precisions.append(precision)
            recalls.append(recall)
            csmfs.append(csmf_accuracy)
        print('--------------Final results--------------------')
        print("Precision: "+str(statistics.mean(precisions)))
        print("Recall: "+str(statistics.mean(recalls)))    
        print("F1score: "+str(statistics.mean(f1scores)))
        print("Csmf accuracy: "+str(statistics.mean(csmfs)))
        print("Overall it takes " + utils.timeSince(total_start_time))
    else:
        if parameters.PRE_DATA:
            
            X,labels,IDs,X2 = preprocess(input_train)
            testX,testlabels,testIDs,testX2 = preprocess(input_test)
            if 'struct' in parameters.embedding:
                X2,testX2 = utils.combineFeat(X2,testX2,feat)
            
            utils.save_numpy([X,labels,IDs,testX,testlabels,testIDs,X2,testX2],
                             [data_file_X,data_labels,data_ids,data_file_testX,data_testlabels,data_testIDs,data_X2,data_testX2])
        else:
            (X,labels,IDs,testX,testlabels,testIDs,X2,testX2) = utils.load_numpy([data_file_X,data_labels,data_ids,data_file_testX,
                                                                                data_testlabels,data_testIDs,data_X2,data_testX2])
#        print(testX2)
        print(X2.shape,testX2.shape,22222)
        if 'struct' in parameters.embedding:
            def repeat_arr(arr,rep):
                '''
                repeat arrs for larger dimension
                '''
                temp = arr
                for i in range(rep-1):
                    arr = np.hstack((arr,temp))
                return arr
            X2 = repeat_arr(X2,parameters.rep)
            testX2 = repeat_arr(testX2,parameters.rep)
            emb_dim_feat = X2.shape[1]
            print('embeded dimension of structured features: '+str(emb_dim_feat))
            print('X.shape: '+str(X.shape)+';  X2.shape: '+str(X2.shape))
            print('testX.shape: '+str(testX.shape)+';  testX2.shape: '+str(testX2.shape))
        all_labels = np.concatenate([labels,testlabels])
        labenc = labelencoder
        labenc.fit(all_labels)
        Y = labenc.transform(labels)
        Y = to_categorical(Y)
#        print(labenc.transform(labels))
        print('Y.shape: '+str(Y.shape))
        all_categories = list(set(list(all_labels)))
        print("all categories: "+str(all_categories))
        class_num = len(all_categories)
        
        if parameters.TRAINING:
            if parameters.embedding == 'conc':
                print('embedding: combination of word and character embedding at input level')
                model = model_format.CNN_TEXT(emb_dim_comb,class_num,dropout=0.0, kernel_sizes=5)
                model.fit(X,Y,emb_dim_comb,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'elmo':
                print('embedding: pubMed elmo embedding')
                emb_dim = 1024
                model = model_format.CNN_ELMO(emb_dim,class_num,USE_SERVER=USE_SERVER)
                model.fit(X,Y,num_epochs=parameters.num_epochs,batch_size=parameters.batch_size,
                          learning_rate=parameters.learning_rate,dropout=parameters.dropout)
            elif parameters.embedding == 'char':
                model = model_format.CNN_TEXT(emb_dim_char,class_num,dropout=0.0, kernel_sizes=5,USE_SERVER=USE_SERVER)
                model.fit(X,Y,emb_dim_char,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'word':
                model = model_format.CNN_TEXT(emb_dim_word,class_num,dropout=0.0, kernel_sizes=5,USE_SERVER=USE_SERVER)
                model.fit(X,Y,emb_dim_word,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'struct':
                print('embedding: narrative data with structured features')
                model = model_format.CNN_STRUCT(emb_dim_comb,emb_dim_feat,class_num,dropout=0.0, kernel_sizes=5,USE_SERVER=parameters.USE_SERVER)
                model.fit(X,X2,Y,emb_dim_comb,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'struct-elmo':
                emb_dim = 1024
                print('embedding: narrative data with structured features with elmo')
                model = model_format.STRUCT_ELMO(emb_dim,emb_dim_feat,class_num,dropout=0.0, kernel_sizes=5, USE_SERVER=USE_SERVER)
                model.fit(X,X2,Y,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'struct-bert':
                print('bert embedding with structured features')
                tmp_d = torch.load(parameters.BERT_WEIGHTS, map_location=cuda)
                state_dict = OrderedDict()
                for i in list(tmp_d.keys())[:199]:
                    x = i
                    if i.find('bert') > -1:
                        x = '.'.join(i.split('.')[1:])
                    state_dict[x] = tmp_d[i]
                # Define model 
                config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
                bert = BertModel(config)
                bert.load_state_dict(state_dict)
#                bert = BertModel.from_pretrained('bert-base-uncased')
                bert.eval()   
                
                model = model_format.STRUCT_BERT(config.hidden_size,emb_dim_feat,class_num,bert,config,dropout=0.0, 
                                                 kernel_num=config.hidden_size,kernel_sizes=5, USE_SERVER=USE_SERVER)
                model.fit(X,X2,Y,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                          num_epochs=parameters.num_epochs,dropout=parameters.dropout)
            elif parameters.embedding == 'bert':
                print('bert embedding')
                tmp_d = torch.load(parameters.BERT_WEIGHTS, map_location=cuda)
                state_dict = OrderedDict()
                for i in list(tmp_d.keys())[:199]:
                    x = i
                    if i.find('bert') > -1:
                        x = '.'.join(i.split('.')[1:])
                    state_dict[x] = tmp_d[i]
                # Define model 
                config = BertConfig(vocab_size_or_config_json_file=parameters.BERT_CONFIG_FILE)
                bert = BertModel(config)
                bert.load_state_dict(state_dict)
                bert.eval()
                if parameters.model_type == 'cnn':
                    model = model_format.BERT_CNN(config.hidden_size,class_num,bert,config,kernel_num=config.hidden_size)
                    model.fit(X,Y,emb_dim_comb,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                              num_epochs=parameters.num_epochs,dropout=parameters.dropout)
                elif parameters.model_type == 'lstm':
                    model = model_format.BERT_LSTM(config.hidden_size,class_num,bert,config,kernel_num=config.hidden_size,dropout=parameters.dropout)
                    model.fit(X,Y,emb_dim_comb,learning_rate=parameters.learning_rate,batch_size=parameters.batch_size,
                              num_epochs=parameters.num_epochs)
            if not USE_SERVER:   #server disk space is not adequate
                torch.save(model, modelfile)
        else:
            model = torch.load(modelfile)
        y_pred = model.predict(testX,testX2)
        labenc = labelencoder
        predictedlabels = labenc.inverse_transform(y_pred)
        print('--------------Final results--------------------')
        precision, recall, f1score, csmf_accuracy = utils.stats_from_results(testlabels,predictedlabels,testIDs,PRINT=True)
        print("Overall it takes " + utils.timeSince(total_start_time))

def preprocess(input_train,ind=0,X2=[]):
    '''
    INPUT:
        input_train: path to the training xml file
    OUTPUT:
        X: a list of strings, containing the narratives
        ID: a list of strings, containing MG_IDs
        label: numpy array of string, containing the labels
    '''
    if parameters.dataset == 'phmrc':
        def csv_to_dic(file):
            with open(file, mode='r') as infile:
                reader = csv.reader(infile)
                mydict = {rows[0]:rows[1] for rows in reader}
            return mydict
        label_dic = csv_to_dic("D:/projects/zhaodong/research/va/data/phmrc_data/map_phmrc_to_cghr_adult.csv")
        print('label dictionary: '+str(label_dic))
        data,all_categories = utils.get_data_phmrc(input_train,label_dic)
    else:
        data,all_categories = utils.get_data_struct(input_train,parameters.feat)
    if parameters.embedding == 'elmo':
        from allennlp.modules.elmo import batch_to_ids
        ID = []
        label = []
        sentence = []
        for k,v in data.items():
            ID.append(k)
            label.append(v[0])
            text = v[1].split()
            sentence.append(text)
        label = np.array(label)
        X = batch_to_ids(sentence)   
        del sentence
        if X.size(1) > max_num_word:
            X = X[:,:max_num_word,:]
        else:
            p2d = (0, 0, 0, max_num_word-X.size(1)) 
            X = F.pad(X, p2d, 'constant', 0)
        if parameters.DEBUG:
            print(type(X))
    elif parameters.embedding == 'struct-elmo':
        from allennlp.modules.elmo import batch_to_ids
        ID = []
        label = []
        sentence = []
        X2 = []
        for k,v in data.items():
            ID.append(k)
            label.append(v[0])
            text = v[1].split()
            sentence.append(text)
            if type(feat) == list:
                temp_list = []
                for f in feat:
                    if v[2][f] == None:
                        temp_list.append(0)
                    else:
                        temp_list.append(v[2][f])
                X2.append(temp_list)
            else:
                if v[2] == None:
                    X2.append(0)
                else:
                    X2.append(v[2]) 
#        print(len(X2))
        label = np.array(label)
        X = batch_to_ids(sentence)   
        del sentence
        if X.size(1) > max_num_word:
            X = X[:,:max_num_word,:]
        else:
            p2d = (0, 0, 0, max_num_word-X.size(1)) 
            X = F.pad(X, p2d, 'constant', 0)

   
    elif parameters.embedding == 'conc':
        char_fname = 'char_emb/code/char_emb_'+str(emb_dim_char)+'.txt'
        vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
        emb_dic = utils.get_dic(char_fname,vocab)
        if parameters.CROSS_VAL:
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
                text = v[i][1]
                if 'stem' in parameters.preprocess:
                    text = preprocessing.stem(text)
                if 'lemmatize' in parameters.preprocess:
                    text = preprocessing.lemmatize(text)
                X.append(lineToNumpy(v[i][1]))
                label.append(k)
    elif parameters.embedding == 'char':
        char_fname = 'char_emb/code/char_emb_'+str(emb_dim_char)+'.txt'
        vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
        emb_dic = utils.get_dic(char_fname,vocab)
        def letterToNumpy(letter):
            return np.array(emb_dic[letter])
        def lineToNumpy(line):
            emb_line = np.zeros((max_num_char,emb_dim_char))
            for i in range(max_num_char):
                if i < len(line):
                    emb_line[i,:] = letterToNumpy(line[i])
            return emb_line
#        data,all_categories = utils.get_data_struct(input_train,features)
        ID = []
        X,label = [],[]
        for k,v in data.items():
            for i in range(len(v)):
                ID.append(k)
                X.append(lineToNumpy(v[i][1]))
                label.append(v[i][0])
    elif parameters.embedding == 'word':
#        print(data['c1587'],11)
        if parameters.CROSS_VAL:
            wmodel,dim = load('D:/projects/zhaodong/research/va/data/crossval_sets/ice+medhelp+narr_all_'+str(ind)+'.vectors.100')
        else:
            if parameters.USE_SERVER:
                wmodel,dim = load('/u/yanzhaod/data/narr+ice+medhelp.vectors.100')
            else:
                wmodel,dim = load('D:/projects/zhaodong/research/va/data/dataset/narr+ice+medhelp.vectors.100')
        def lineToNumpyWord(line):
            '''
            INPUT:
                line: string
            OUTPUT:
                emb_lin: numpy array of size (max_num_word,emb_dim_comb)
            '''
            l = line.split(' ')
            if 'backward_padding' in parameters.preprocess:
#                if len(l) < parameters.max_num_word:
#                    for i in range(len(l)):
#                        emb_line.append(np.array(get(l[i],wmodel)))
#                    pad = np.zeros((parameters.max_num_word-len(l),parameters.emb_dim_word))
#                    e
#                
#                    for i in range(len(l)):
#                        emb_line.append(list(get(l[i],wmodel)))
#                    pad = [0]*parameters.emb_dim_word
#                    emb_line = [pad]*(parameters.max_num_word-len(l))+emb_line
                    
                emb_line = np.zeros((min(parameters.max_num_word,(len(l))),parameters.emb_dim_word))
                for i in range(emb_line.shape[0]):
                    emb_line[i,:] = np.array(get(l[i],wmodel))
                if emb_line.shape[0] < parameters.max_num_word:
                    emb_line  = np.concatenate((np.zeros((parameters.max_num_word-emb_line.shape[0],parameters.emb_dim_word)),emb_line),axis=0)
            else:
                emb_line = np.zeros((parameters.max_num_word,parameters.emb_dim_word))
                for i in range(max_num_word):
                    if i < len(l):
                        emb_line[i,:] = np.array(get(l[i],wmodel))
            return emb_line
        ID = []
        X,label = [],[]
        X2 = []
#        print(data['c1587'],2)
#        print(data.keys())
        for k,v in data.items():
            ID.append(k)
            X.append(lineToNumpyWord(v[1]))
            label.append(v[0])
    elif parameters.embedding == 'struct':
        char_fname = 'char_emb/code/char_emb_'+str(emb_dim_char)+'.txt'
        vocab='abcdefghijklmnopqrstuvwxyz0123456789 '
        emb_dic = utils.get_dic(char_fname,vocab)
        if parameters.CROSS_VAL:
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
        X2 = []
        for k,v in data.items():
            for i in range(len(v)):
                ID.append(v[i][0])
                X.append(lineToNumpy(v[i][1]))
                label.append(k)
                if type(feat) == list:
                    temp_list = []
                    for f in feat:
                        if v[i][2][f] == None:
                            temp_list.append(0)
                        else:
                            temp_list.append(v[i][2][f])
                    X2.append(temp_list)
                else:
                    if v[i][2] == None:
                        X2.append(0)
                    else:
                        X2.append(v[i][2])                    
    elif parameters.embedding == 'bert':
        tokenizer = BertTokenizer(vocab_file=parameters.VOCAB_FILE, do_lower_case=True)
                
        sents,label = [],[]    
        ID = []
        for k,v in data.items():
            ID.append(k)
            sents.append(lineToNumpyWord(v[1]))
            label.append(v[0])
#        for k,v in data.items():
#            for i in range(len(v)):
#                ID.append(v[i][0])
#                sents.append(v[i][1])
#                label.append(k)
        X = np.zeros((len(sents),max_num_word),dtype=np.int)
        for i,ids in tqdm(enumerate(sents)):
            tokens = tokenizer.tokenize(ids.lower())[:max_num_word-2]
#            tokens = tokenizer.tokenize(ids)[:max_num_word-2]
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens+["[SEP]"])
            inp_len = len(input_ids)
            if inp_len>max_num_word:
                X[i,:max_num_word] = np.array(input_ids)[:max_num_word]
            else:
                X[i,:inp_len] = np.array(input_ids)
    elif parameters.embedding == 'struct-bert':
        tokenizer = BertTokenizer(vocab_file=parameters.VOCAB_FILE, do_lower_case=True)
                
        sents,label = [],[]    
        ID = []
        X2 = []
        for k,v in data.items():
            for i in range(len(v)):
                ID.append(v[i][0])
                label.append(k)
                text = v[i][1].split()
                sents.append(v[i][1])
                if type(feat) == list:
                    temp_list = []
                    for f in feat:
                        if v[i][2][f] == None:
                            temp_list.append(0)
                        else:
                            temp_list.append(v[i][2][f])
                    X2.append(temp_list)
                else:
                    if v[i][2] == None:
                        X2.append(0)
                    else:
                        X2.append(v[i][2]) 
        X = np.zeros((len(sents),max_num_word),dtype=np.int)
        for i,ids in tqdm(enumerate(sents)):
            tokens = tokenizer.tokenize(ids.lower())[:max_num_word-2]
#            tokens = tokenizer.tokenize(ids)[:max_num_word-2]
            input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens+["[SEP]"])
            inp_len = len(input_ids)
            if inp_len>max_num_word:
                X[i,:max_num_word] = np.array(input_ids)[:max_num_word]
            else:
                X[i,:inp_len] = np.array(input_ids)
        del sents
    else:
        print('the embedding type %s is not recognized'%parameters.embedding)
    X2 = np.asarray(X2).astype('float')
    X = np.asarray(X).astype('float')
#    print(X2.shape,2222)
    return X,label,ID,X2

if __name__ == "__main__":main() 
