#!/usr/bin/python
# Neural network model functions
# @author sjeblee@cs.toronto.edu

#import sys
#sys.path.append('../keras-attention-mechanism')

import argparse
import numpy
import os
import time
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Permute, Reshape, RepeatVector, BatchNormalization
from keras.layers import Embedding, LSTM, GRU, TimeDistributed, Merge, merge, concatenate, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
#from numpy import array, int32

#import attention_utils
import data_util
import rebalance
from layers import Attention

vec_types = ["narr_vec", "narr_seq", "event_vec", "event_seq"]
numpy.set_printoptions(threshold=numpy.inf)

''' Create and train a feed-forward neural network model. Features selection should be done before passing to this function
    X: a python list or numpy array of training data of shape [num_samples, num_features]
    Y: a python list or numpy array of training labels of shape [num_samples]
    returns: the model and the modified X and Y arrays
'''
def nn_model(X, Y, num_nodes, act, num_epochs=10):
    X = numpy.asarray(X)
    Y = to_categorical(Y)
    print "X.shape: " + str(X.shape)
    print "Y.shape: " + str(Y.shape)

    print "neural network: nodes: " + str(num_nodes)
    nn = Sequential([Dense(num_nodes, input_dim=X.shape[-1]),
                    Activation(act),
                    #Dense(num_nodes, input_dim=num_feats),
                    #Activation(activation),
                    Dense(Y.shape[1]),
                    Activation('softmax'),])
        
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(X, Y, epochs=num_epochs)
    nn.summary()
    return nn, X, Y

''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_model(X, Y, num_nodes, activation='sigmoid', modelname='lstm', dropout=0.1, X2=[], pretrainX=[], pretrainY=[], pretrainX2=[], initial_states=None, num_epochs=15):
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    inputs = []
    input_arrays = [X]
    hybrid = False
    pretrain = False
    if len(pretrainX) > 0 and len(pretrainY) > 0:
        pretrain = True
        print "Using pretraining"

    print "model: " + modelname + " nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)

    input_shape = (max_seq_len, embedding_size)
    input1 = Input(shape=input_shape)
    inputs.append(input1)

    # Handle hybrid model
    if len(X2) > 0:
        hybrid = True
        X2 = numpy.asarray(X2)
        print "X2 shape: " + str(X2.shape)
        input_arrays.append(X2)
        ff_feats = X2.shape[1]
        input2 = Input(shape=(ff_feats,))
        inputs.append(input2)
        ff_feats = 100
        ff = Dense(ff_feats, activation='relu')(input2)

    if modelname == 'rnn':
        rnn = SimpleRNN(num_nodes, return_sequences=False, return_state=True)
    if modelname == 'gru':
        rnn = GRU(num_nodes, return_sequences=False, return_state=True)
    else:
        rnn = LSTM(num_nodes, return_sequences=False, return_state=True)

    if initial_states == None:
        rnn_out, rnn_states = rnn(input1)
    else:
        rnn_out, rnn_states = rnn(input1, initial_state=initial_states)
    dropout_out = Dropout(dropout)(rnn_out)
    #attn_out = attention(dropout_out, max_seq_len, embedding_size)

    if hybrid:
        #print "ff shape: " + str(ff.output_shape)
        last_out = concatenate([dropout_out, ff], axis=-1)
    else:
        last_out = dropout_out

    if pretrain:
        print "pretraining..."
        for k in range(len(pretrainX)):
            trainX = numpy.asarray(pretrainX[k])
            trainY = to_categorical(pretrainY[k])
            pretrain_input_arrays = [trainX]
            pretrain_inputs = [Input(shape=input_shape)]
            if hybrid:
                trainX2 = numpy.asarray(pretrainX2[k])
                pretrain_input_arrays.append(trainX2)
                                        
            prediction = Dense(trainY.shape[1], activation='softmax')(last_out)
            pre_nn = Model(inputs=inputs, outputs=prediction)
            pre_nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            pre_nn.fit(pretrain_input_arrays, trainY, epochs=num_epochs)

    print "training with main data..."
    prediction = Dense(Y.shape[1], activation='softmax')(last_out)
    nn = Model(inputs=inputs, outputs=prediction)
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(input_arrays, Y, epochs=num_epochs)

    nn.summary()
    return nn, X, Y

''' Create and train a CNN model
    Hybrid features supported - pass structured feats as X2
    Does NOT support joint training yet
    returns: the CNN model
'''
def cnn_model(X, Y, act=None, windows=[1,2,3,4,5], X2=[], num_epochs=10, return_layer=False):
    #Y = to_categorical(Y)
    X = numpy.asarray(X)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    print "train X shape: " + str(X.shape)
    print "CNN: embedding: " + str(embedding_size)
    print "max_seq_len: " + str(max_seq_len)
    branches = []
    inputs = []
    input_arrays = [X]
    hybrid = False

    # Keras functional API with attention
    # Input layers
    input_shape = (max_seq_len, embedding_size) 
    input1 = Input(shape=input_shape)
    inputs.append(input1)
    if len(X2) > 0:
        hybrid = True
        X2 = numpy.asarray(X2)
        print "X2 shape: " + str(X2.shape)
        input_arrays.append(X2)
        input2 = Input(shape=(X2.shape[1],))
        inputs.append(input2)
        ff = Dense(10, activation='relu')(input2)

    # Attention
    #attn_out = attention(inputs, max_seq_len, embedding_size)

    # Convolution
    conv_outputs = []
    for w in windows:
        print "window: " + str(max_seq_len) + " x " + str(w)
        conv_layer = Conv1D(max_seq_len, w, input_shape=input_shape)
        conv = conv_layer(input1)
        max_pool_layer = GlobalMaxPooling1D()
        max_pool = max_pool_layer(conv)
        conv_outputs.append(max_pool)
        print "conv: " + str(conv_layer.output_shape) + " pool: " + str(max_pool_layer.output_shape)

    # Merge
    merged = concatenate(conv_outputs, axis=-1)
    #print "conv shape: " + str(merged.output_shape)
    if hybrid:
        #print "ff shape: " + str(ff.output_shape)
        merged = concatenate([merged, ff], axis=-1)

    # Prediction
    prediction = Dense(Y.shape[1], activation='softmax')(merged)
    nn = Model(inputs=inputs, outputs=prediction)

    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(input_arrays, Y, epochs=num_epochs)
    nn.summary()

    #try:
    #    print "attention weights: " + str(attention_layer.get_weights())
    #except AttributeError:
    #    print "ERROR: got an exception trying to print attention weights"

    if return_layer:
        return nn, prediction
    else:
        return nn, X, Y

''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_keyword_model(X, Y, num_nodes, activation='sigmoid', modelname='lstm', dropout=0.1, X2=[], pretrainX=[], pretrainY=[], pretrainX2=[], keywords=[], initial_states=None, windows=[1,2,3,4,5], num_epochs=15):
    print "keyword rnn"
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    keywords = numpy.asarray(keywords)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    inputs = []
    input_arrays = [X]
    hybrid = False
    pretrain = False

    input_shape = (max_seq_len, embedding_size)
    input1 = Input(shape=input_shape)
    inputs.append(input1)
    
    # Keyword CNN
    #conv_outputs = []
    #for w in windows:
    #    print "window: " + str(max_seq_len) + " x " + str(w)
    #    conv_layer = Conv1D(max_seq_len, w, input_shape=input_shape)
    #    conv = conv_layer(input1)
    #    max_pool_layer = GlobalMaxPooling1D()
    #    max_pool = max_pool_layer(conv)
    #    conv_outputs.append(max_pool)
    #    print "conv: " + str(conv_layer.output_shape) + " pool: " + str(max_pool_layer.output_shape)
    #merged = concatenate(conv_outputs, axis=-1)

    # Keyword GRU
    kw_rnn = GRU(num_nodes, return_sequences=False, return_state=False)
    kw_rnn_out = kw_rnn(input1)
    kw_out = Dropout(dropout)(kw_rnn_out)

    kw_prediction = Dense(keywords.shape[1], activation='softmax')(kw_out)
    kw_model = Model(inputs=inputs, outputs=kw_prediction)

    kw_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    kw_model.fit(input_arrays, keywords, epochs=num_epochs)
    kw_model.summary()

    # RNN Pretraining
    if len(pretrainX) > 0 and len(pretrainY) > 0:
        pretrain = True
        print "Using pretraining"

    print "model: " + modelname + " nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)

    # Handle hybrid model
    if len(X2) > 0:
        print "hybrid not supported yet!"
        #hybrid = True
        #X2 = numpy.asarray(X2)
        #print "X2 shape: " + str(X2.shape)
        #input_arrays.append(X2)
        #ff_feats = X2.shape[1]
        #input2 = Input(shape=(ff_feats,))
        #inputs.append(input2)
        #ff = Dense(ff_feats, activation='relu')(input2)

    if modelname == 'rnn':
        rnn = SimpleRNN(num_nodes, return_sequences=False, return_state=True)
    if modelname == 'gru':
        rnn = GRU(num_nodes, return_sequences=False, return_state=True)
    else:
        rnn = LSTM(num_nodes, return_sequences=False, return_state=True)

    if initial_states == None:
        rnn_out, rnn_states = rnn(input1)
    else:
        rnn_out, rnn_states = rnn(input1, initial_state=initial_states)
    dropout_out = Dropout(dropout)(rnn_out)
    #attn_out = attention(dropout_out, max_seq_len, embedding_size)

    #if hybrid:
        #print "ff shape: " + str(ff.output_shape)
    #    last_out = concatenate([dropout_out, ff], axis=-1)
    #else:
    #    last_out = dropout_out

    kw_pred_main = Dense(keywords.shape[1], activation='softmax')(kw_out)
    last_out = concatenate([dropout_out, kw_pred_main], axis=-1)

    if pretrain:
        print "pretraining..."
        for k in range(len(pretrainX)):
            trainX = numpy.asarray(pretrainX[k])
            trainY = to_categorical(pretrainY[k])
            pretrain_input_arrays = [trainX]
            pretrain_inputs = [Input(shape=input_shape)]
            if hybrid:
                trainX2 = numpy.asarray(pretrainX2[k])
                pretrain_input_arrays.append(trainX2)

            prediction = Dense(trainY.shape[1], activation='softmax')(last_out)
            pre_nn = Model(inputs=inputs, outputs=prediction)
            pre_nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            pre_nn.fit(pretrain_input_arrays, trainY, epochs=num_epochs)

    print "training with main data..."
    prediction = Dense(Y.shape[1], activation='softmax')(last_out)
    nn = Model(inputs=inputs, outputs=prediction)
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(input_arrays, Y, epochs=num_epochs)

    nn.summary()
    return nn, kw_model, X, Y

''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_cnn_model(X, Y, num_nodes, activation='sigmoid', modelname='lstm', dropout=0.1, X2=[], pretrainX=[], pretrainY=[], pretrainX2=[], initial_states=None, windows=[1,2,3,4,5], num_epochs=15):
    print "rnn cnn model"
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    inputs = []
    input_arrays = [X]
    hybrid = False
    pretrain = False

    input_shape = (max_seq_len, embedding_size)
    input1 = Input(shape=input_shape)
    inputs.append(input1)

    # CNN
    #conv_outputs = []
    #for w in windows:
    #    print "window: " + str(max_seq_len) + " x " + str(w)
    #    conv_layer = Conv1D(max_seq_len, w, input_shape=input_shape)
    #    conv = conv_layer(input1)
    #    max_pool_layer = GlobalMaxPooling1D()
    #    max_pool = max_pool_layer(conv)
    #    conv_outputs.append(max_pool)
    #    print "conv: " + str(conv_layer.output_shape) + " pool: " + str(max_pool_layer.output_shape)
    #pre_out = concatenate(conv_outputs, axis=-1)

    # GRU
    pre_rnn = GRU(num_nodes, return_sequences=False, return_state=False)
    pre_rnn_out = pre_rnn(input1)
    pre_out = Dropout(dropout)(pre_rnn_out)

    #pre_prediction = Dense(keywords.shape[1], activation='softmax')(kw_out)
    #kw_model = Model(inputs=inputs, outputs=kw_prediction)

    #kw_model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #kw_model.fit(input_arrays, keywords, epochs=num_epochs)
    #kw_model.summary()

    # RNN Pretraining
    if len(pretrainX) > 0 and len(pretrainY) > 0:
        pretrain = True
        print "Using pretraining"

    print "model: " + modelname + " nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)

    # Handle hybrid model
    if len(X2) > 0:
        print "hybrid not supported yet!"
        #hybrid = True
        #X2 = numpy.asarray(X2)
        #print "X2 shape: " + str(X2.shape)
        #input_arrays.append(X2)
        #ff_feats = X2.shape[1]
        #input2 = Input(shape=(ff_feats,))
        #inputs.append(input2)
        #ff = Dense(ff_feats, activation='relu')(input2)

    if modelname == 'rnn':
        rnn = SimpleRNN(num_nodes, return_sequences=False, return_state=True)
    if modelname == 'gru':
        rnn = GRU(num_nodes, return_sequences=False, return_state=True)
    else:
        rnn = LSTM(num_nodes, return_sequences=False, return_state=True)

    if initial_states == None:
        rnn_out, rnn_states = rnn(input1)
    else:
        rnn_out, rnn_states = rnn(input1, initial_state=initial_states)
    dropout_out = Dropout(dropout)(rnn_out)
    #attn_out = attention(dropout_out, max_seq_len, embedding_size)

    #if hybrid:
        #print "ff shape: " + str(ff.output_shape)
    #    last_out = concatenate([dropout_out, ff], axis=-1)
    #else:
    #    last_out = dropout_out

    #kw_pred_main = Dense(keywords.shape[1], activation='softmax')(kw_out)
    last_out = concatenate([dropout_out, pre_out], axis=-1)

    if pretrain:
        print "pretraining..."
        for k in range(len(pretrainX)):
            trainX = numpy.asarray(pretrainX[k])
            trainY = to_categorical(pretrainY[k])
            pretrain_input_arrays = [trainX]
            pretrain_inputs = [Input(shape=input_shape)]
            if hybrid:
                trainX2 = numpy.asarray(pretrainX2[k])
                pretrain_input_arrays.append(trainX2)

            prediction = Dense(trainY.shape[1], activation='softmax')(last_out)
            pre_nn = Model(inputs=inputs, outputs=prediction)
            pre_nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            pre_nn.fit(pretrain_input_arrays, trainY, epochs=num_epochs)

    print "training with main data..."
    prediction = Dense(Y.shape[1], activation='softmax')(last_out)
    nn = Model(inputs=inputs, outputs=prediction)
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(input_arrays, Y, epochs=num_epochs)

    nn.summary()
    return nn, X, Y

''' TODO: Fix this
'''
def create_filter_rnn_model(X, Y, embedding_size, num_nodes, activation='tanh', dropout=0.1, hybrid=False, X2=[]):
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
    max_seq_len = X.shape[1]
    inputs = []
    input_arrays = [X]
    num_nodes = 56
    num_epochs = 30
    dropout = 0.2

    print "GRU: nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)
    input_shape = (max_seq_len, embedding_size)
    input1 = Input(shape=input_shape)
    inputs.append(input1)
    #if hybrid:
    #    X2 = numpy.asarray(X2)
    #    print "X2 shape: " + str(X2.shape)
    #    input_arrays.append(X2)
    #    input2 = Input(shape=(X2.shape[1],))
    #    inputs.append(input2)
    #    ff = Dense(10, activation='relu')(input2)

    rnn1 = GRU(num_nodes, return_sequences=True)
    rnn1_out = rnn1(input1)
    print "rnn1 output_shape: " + str(rnn1.output_shape)
    dense0_out = TimeDistributed(Dense(num_nodes, activation='tanh'))(rnn1_out)
    dense1 = TimeDistributed(Dense(1, activation='softmax'), name='dense1')
    weights = dense1(dense0_out)
    print "dense1 output_shape: " + str(dense1.output_shape)
    #norm_weights = BatchNormalization(name='norm_weights')
    #norm_weights_out = norm_weights(weights) # TODO: normalize values to 0 to 1

    # Repeat the weights across embedding dimensions
    #permute1 = Permute((2,1))
    #permuted_weights = permute1(norm_weights)
    #print "permute1 output_shape: " + str(permute1.output_shape)
    
    repeat = TimeDistributed(RepeatVector(embedding_size))
    repeated_weights = repeat(weights)

    print "repeat input_shape: " + str(repeat.input_shape)
    print "repeat output_shape: " + str(repeat.output_shape)
    final_weights = Reshape(input_shape)(repeated_weights)
    #TODO: mutiply layer - need to convert 1d weight to embedding_size vector?
    filter_out = multiply([input1, final_weights])
    #filter_out = merge([input1, norm_weights], mode=scalarMult, output_shape=input_shape) 
    #TODO: masking layer???
    #Masking(mask_value) # but want <= mask_value, not just ==
    # TODO: save the weights for each word so we can look at them

    rnn2_out = GRU(num_nodes, return_sequences=False)(filter_out)
    dropout_out = Dropout(dropout)(rnn2_out)

    #if hybrid:
    #   print "ff shape: " + str(ff.output_shape)
    #    merged = concatenate([dropout_out, ff], axis=-1)

    prediction = Dense(Y.shape[1], activation='softmax')(dropout_out)
    nn = Model(inputs=inputs, outputs=prediction)

    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    nn.fit(input_arrays, Y, epochs=num_epochs)
    nn.summary()

    # Save weights
    print "saving weights for train data"
    weight_model = Model(inputs=inputs, outputs=nn.get_layer('dense1').output)
    train_weights = weight_model.predict(X)
    filename = "filternn_weights"
    outfile = open(filename, 'w')
    outfile.write(str(train_weights))
    outfile.close()
    
    return nn, X, Y    

def scalarMult(layersList):
    vector = layersList[0]
    scalar = layersList[1]
    return vector * scalar

def attention(inputs, time_steps, input_dim):
    #input_dim = int(inputs.shape[-1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul
