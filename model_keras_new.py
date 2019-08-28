# -*- coding: utf-8 -*-
"""
Created on Sun May  5 08:10:00 2019

@author: Zhaodong Yan
"""

import numpy as np 
import pandas as pd 
from tqdm import tqdm, trange
import pickle
import gc
import time
import preprocessing
from collections import OrderedDict
from pytorch_pretrained_bert.modeling import BertConfig
BERT_MODEL = 'bert-base-uncased'
CASED = 'uncased' in BERT_MODEL
INPUT = '../input/jigsaw-bert-preprocessed-input/'
TEXT_COL = 'comment_text'

from pytorch_pretrained_bert import BertTokenizer
from pytorch_pretrained_bert.modeling import BertModel

BERT_FP = 'bert-base-uncased'
DOWNLOAD_DATA = True   #otherwise directly load data
TRAIN_MODEL = True      #otherwise directly load model


import parameters
import utils
import sklearn
import torch
labelencoder = sklearn.preprocessing.LabelEncoder()
cuda = torch.device("cuda:0")
from keras.utils.np_utils import to_categorical
def get_bert_embed_matrix():
#    bert = BertModel.from_pretrained(BERT_FP)
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
    
    bert_embeddings = list(bert.children())[0]
    bert_word_embeddings = list(bert_embeddings.children())[0]
    mat = bert_word_embeddings.weight.data.numpy()
    return mat

st = time.time()
embedding_matrix = get_bert_embed_matrix()
from keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, Dropout, add, concatenate
from keras.layers import CuDNNLSTM, CuDNNGRU, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.preprocessing import text, sequence
from keras.losses import binary_crossentropy
from keras import backend as K
import keras.layers as L
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers

from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import KFold
from keras.models import load_model
from keras import backend as K

import statistics
#class Attention(Layer):
#    def __init__(self, step_dim,
#                 W_regularizer=None, b_regularizer=None,
#                 W_constraint=None, b_constraint=None,
#                 bias=True, **kwargs):
#        self.supports_masking = True
#        self.init = initializers.get('glorot_uniform')
#
#        self.W_regularizer = regularizers.get(W_regularizer)
#        self.b_regularizer = regularizers.get(b_regularizer)
#
#        self.W_constraint = constraints.get(W_constraint)
#        self.b_constraint = constraints.get(b_constraint)
#
#        self.bias = bias
#        self.step_dim = step_dim
#        self.features_dim = 0
#        super(Attention, self).__init__(**kwargs)
#
#    def build(self, input_shape):
#        assert len(input_shape) == 3
#
#        self.W = self.add_weight((input_shape[-1],),
#                                 initializer=self.init,
#                                 name='{}_W'.format(self.name),
#                                 regularizer=self.W_regularizer,
#                                 constraint=self.W_constraint)
#        self.features_dim = input_shape[-1]
#
#        if self.bias:
#            self.b = self.add_weight((input_shape[1],),
#                                     initializer='zero',
#                                     name='{}_b'.format(self.name),
#                                     regularizer=self.b_regularizer,
#                                     constraint=self.b_constraint)
#        else:
#            self.b = None
#
#        self.built = True
#
#    def compute_mask(self, input, input_mask=None):
#        return None
#
#    def call(self, x, mask=None):
#        features_dim = self.features_dim
#        step_dim = self.step_dim
#
#        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
#                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))
#
#        if self.bias:
#            eij += self.b
#
#        eij = K.tanh(eij)
#
#        a = K.exp(eij)
#
#        if mask is not None:
#            a *= K.cast(mask, K.floatx())
#
#        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())
#
#        a = K.expand_dims(a)
#        weighted_input = x * a
#        return K.sum(weighted_input, axis=1)
#
#    def compute_output_shape(self, input_shape):
#        return input_shape[0],  self.features_dim
def build_model(embedding_matrix):
    '''
    credits go to: https://www.kaggle.com/thousandvoices/simple-lstm/
    '''
    words = Input(shape=(parameters.max_num_word,))
    feats = Input(shape=(X2.shape[1],))
    x = Embedding(*embedding_matrix.shape, weights=[embedding_matrix], trainable=True)(words)
    
    x = SpatialDropout1D(parameters.dropout)(x)
    if parameters.model_type == 'lstm':
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNLSTM(LSTM_UNITS, return_sequences=True))(x)
    elif parameters.model_type == 'gru':
#        x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)
        x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)
#        x = Bidirectional(CuDNNGRU(LSTM_UNITS, return_sequences=True))(x)
    else:
        print('ERROR: model must be "lstm" or "gru"')
#    print(x.shape,2321313)
#    x = Attention(x.shape[1])(x)
    if 'attention' in parameters.module:
        x = Attention(x.shape[1])(x)
        hidden = concatenate([x,feats])
    else:
        hidden = concatenate([GlobalMaxPooling1D()(x),GlobalAveragePooling1D()(x),])
        hidden = add([hidden, Dense(DENSE_HIDDEN_UNITS, activation='relu')(hidden)])
        
        hidden = concatenate([hidden,feats])

    combined = Dense(class_num, activation='sigmoid')(hidden)
    
    model = Model(inputs=[words,feats], outputs=combined)
    
    model.compile(loss='categorical_crossentropy', loss_weights= [1.0], optimizer='adam')
    return model
def preprocess(input_data):
    data,all_categories = utils.get_data_struct(input_data,parameters.feat)
    tokenizer = BertTokenizer(vocab_file=parameters.VOCAB_FILE, do_lower_case=True)
    sents,label = [],[]    
    ID = []
    X2 = []
    for k,v in data.items():
        ID.append(k)
        label.append(v[0])
        text = v[1]
        if 'stem' in parameters.preprocess:
            text = preprocessing.stem(text)
        if 'lemmatize' in parameters.preprocess:
            text = preprocessing.lemmatize(text)
        sents.append(text)
        if type(parameters.feat) == list:
            temp_list = []
            for f in parameters.feat:
                if v[2][f] == None:
                    temp_list.append(0)
                else:
                    temp_list.append(v[2][f])
            X2.append(temp_list)
        else:
            if v[i][2] == None:
                X2.append(0)
            else:
                X2.append(v[i][2]) 
    X = np.zeros((len(sents),parameters.max_num_word),dtype=np.int)
    for i,ids in tqdm(enumerate(sents)):
    #    try:
        tokens = tokenizer.tokenize(ids.lower())[:parameters.max_num_word-2]
        input_ids = tokenizer.convert_tokens_to_ids(["[CLS]"]+tokens+["[SEP]"])
        inp_len = len(input_ids)
        X[i,:inp_len] = np.array(input_ids)
    X2 = np.asarray(X2).astype('float')
    return X,label,ID,X2
if parameters.CROSS_VAL:
    f1scores,precisions,recalls,csmfs = [],[],[],[]
    for i in range(10):
        input_train = "D:/projects/zhaodong/research/va/data/crossval_sets/train_"+parameters.rec_type+"_"+str(i)+"_cat_spell.xml"  #input train file for char_embeeding
        input_test = "D:/projects/zhaodong/research/va/data/crossval_sets/test_"+parameters.rec_type+"_"+str(i)+"_cat_spell.xml"    #input test file for char_embedding
        X,labels,IDs,X2 = preprocess(input_train)
        testX,testlabels,testIDs,testX2 = preprocess(input_test)
    #    print(X2.shape,testX2.shape,12312323)
        X2,testX2 = utils.combineFeat(X2,testX2,parameters.feat)
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
        y_train = labenc.transform(labels)
        y_train = to_categorical(y_train)
        class_num = y_train.shape[1]
        EPOCHS = parameters.num_epochs
        LSTM_UNITS = 32
        DENSE_HIDDEN_UNITS = 128
        model_type = parameters.model_type
        
        model = build_model(embedding_matrix)
        for global_epoch in range(EPOCHS):
            model_name = model_type+'.h5'
            if TRAIN_MODEL:
                model.fit([X,X2],y_train,
                    batch_size=parameters.batch_size,
                    epochs=1,
                    verbose=1,
                    callbacks=[
                        LearningRateScheduler(lambda epoch: 0.01*(0.6**global_epoch))
                    ]
                )    
                model.save(model_name)
            else:
                model = load_model(model_name)
            pred = model.predict([testX,testX2], batch_size=16)
            y_pred = np.argmax(pred,axis=1)
            gc.collect()
        
        gc.collect()
        labenc = labelencoder
        predictedlabels = labenc.inverse_transform(y_pred)
    #    print(len(predictedlabels))
    #    print(len(testlabels))
    #    print(len(testIDs))
        precision, recall, f1score, csmf_accuracy = utils.stats_from_results(testlabels,predictedlabels,testIDs,PRINT=True)
        
        f1scores.append(f1score)
        precisions.append(precision)
        recalls.append(recall)
        csmfs.append(csmf_accuracy)
        gc.collect()
        del X
        del X2
        del all_labels
        #del embedding_matrix
        del labels
        del pred
        del testX
        del testX2
    print('--------------Final results--------------------')
    print("Precision: "+str(statistics.mean(precisions)))
    print("Recall: "+str(statistics.mean(recalls)))    
    print("F1score: "+str(statistics.mean(f1scores)))
    print("Csmf accuracy: "+str(statistics.mean(csmfs)))
    print("Overall it takes " + utils.timeSince(st))
else:
    if parameters.rec_type != 'neonate':
        input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+parameters.rec_type+"_cat_spell.xml"
        input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+parameters.rec_type+"_cat_spell.xml"
    else:
        input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+parameters.rec_type+"_cat.xml"
        input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+parameters.rec_type+"_cat.xml"   
    X,labels,IDs,X2 = preprocess(input_train)
    testX,testlabels,testIDs,testX2 = preprocess(input_test)
#    print(X2.shape,testX2.shape,12312323)
    X2,testX2 = utils.combineFeat(X2,testX2,parameters.feat)
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
    y_train = labenc.transform(labels)
    y_train = to_categorical(y_train)
    class_num = y_train.shape[1]
    EPOCHS = parameters.num_epochs
    LSTM_UNITS = 32
    DENSE_HIDDEN_UNITS = 128
    model_type = parameters.model_type

    
    model = build_model(embedding_matrix)
    for global_epoch in range(EPOCHS):
        model_name = model_type+'.h5'
        if TRAIN_MODEL:
            model.fit([X,X2],y_train,
                batch_size=parameters.batch_size,
                epochs=1,
                verbose=1,
                callbacks=[
                    LearningRateScheduler(lambda epoch: 0.01*(0.6**global_epoch))
                ]
            )    
            model.save(model_name)
        else:
            model = load_model(model_name)
        pred = model.predict([testX,testX2], batch_size=16)
        y_pred = np.argmax(pred,axis=1)
        gc.collect()
    
    gc.collect()
    labenc = labelencoder
    predictedlabels = labenc.inverse_transform(y_pred)
    
    precision, recall, f1score, csmf_accuracy = utils.stats_from_results(testlabels,predictedlabels,testIDs,PRINT=True)
    
    print("Overall it takes " + utils.timeSince(st))
#if parameters.dataset == 'mds':
#    if parameters.rec_type != 'neonate':
#        input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+parameters.rec_type+"_cat_spell.xml"
#        input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+parameters.rec_type+"_cat_spell.xml"
#    else:
#        input_train = "D:/projects/zhaodong/research/va/data/dataset/train_"+parameters.rec_type+"_cat.xml"
#        input_test = "D:/projects/zhaodong/research/va/data/dataset/test_"+parameters.rec_type+"_cat.xml"             
#
#
#prefix = '_'.join([parameters.embedding,parameters.model_type,parameters.rec_type,parameters.dataset,'keras_test'])
#out_dir = 'output/'
#data_file_X = out_dir+prefix+'_train.npy'
#data_file_testX = out_dir+prefix+'_test.npy'
#data_labels = out_dir+prefix+'_labels.npy'
#data_testlabels = out_dir+prefix+'_testlabels.npy'
#data_testIDs = out_dir+prefix+'_testids.npy'
#data_ids = out_dir+prefix+'_ids.npy'
#data_X2 = out_dir+prefix+'_X2.npy'
#data_testX2 = out_dir+prefix+'_testX2.npy'
#modelfile = out_dir+'model/'+prefix+'_model.pt'    # Filename of the saved model

#
#
#
#
#
#if DOWNLOAD_DATA:
#
#    utils.save_numpy([X,labels,IDs,testX,testlabels,testIDs,X2,testX2],
#                     [data_file_X,data_labels,data_ids,data_file_testX,data_testlabels,data_testIDs,data_X2,data_testX2])
#else:
#    (X,labels,IDs,testX,testlabels,testIDs,X2,testX2) = utils.load_numpy([data_file_X,data_labels,data_ids,data_file_testX,
#                                                                        data_testlabels,data_testIDs,data_X2,data_testX2])
#gc.collect()
#if 'struct' in parameters.embedding:
#    def repeat_arr(arr,rep):
#        '''
#        repeat arrs for larger dimension
#        '''
#        temp = arr
#        for i in range(rep-1):
#            arr = np.hstack((arr,temp))
#        return arr
#    X2 = repeat_arr(X2,parameters.rep)
#    testX2 = repeat_arr(testX2,parameters.rep)
#    emb_dim_feat = X2.shape[1]
#    print('embeded dimension of structured features: '+str(emb_dim_feat))
#    print('X.shape: '+str(X.shape)+';  X2.shape: '+str(X2.shape))
#    print('testX.shape: '+str(testX.shape)+';  testX2.shape: '+str(testX2.shape))





#from sklearn.model_selection import train_test_split
#tr_ind, val_ind = train_test_split(list(range(len(X))) ,test_size = 0.05, random_state = 23)



























