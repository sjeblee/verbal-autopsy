#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import sys
sys.path.append('../keras-attention-mechanism')

import argparse
import numpy
import os
import time
from hyperopt import hp, fmin, tpe, space_eval
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Permute, Reshape
from keras.layers import Embedding, LSTM, Merge, merge, concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from numpy import array, int32
from sklearn import metrics
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

import attention_utils
import rebalance

labelencoder = None
vec_types = ["narr_vec", "narr_seq", "event_vec", "event_seq"]

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--model', action="store", dest="model")
    argparser.add_argument('--name', action="store", dest="name")
    argparser.add_argument('--prefix', action="store", dest="prefix")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.testfile and args.model):
        print "usage: python model.py --in [train.features] --test [test.features] --out [test.results] --labels [ICD_cat/ICD_cat_neo/Final_code] --model [nn] --name [rnn_ngram3] --prefix [/sjeblee/research/models]"
        exit()

    labelname = "Final_Code"
    if args.labelname:
        labelname = args.labelname

    prefix = None
    if args.prefix:
        prefix = args.prefix

    run(args.model, args.name, args.infile, args.testfile, args.outfile, prefix, labelname)

def hyperopt(arg_model, arg_train_feats, arg_test_feats, arg_result_file, arg_prefix, arg_labelname):
    print "hyperopt"

    # Set up data file references
    global h_model, h_train, h_test, h_result, h_prefix, labelname
    h_model = arg_model
    h_train = arg_train_feats
    h_test = arg_test_feats
    h_result = arg_result_file
    h_prefix = arg_prefix
    labelname = arg_labelname

    global n_feats
    n_feats = 200
    space = None
    objective = None

    print "h_model: " + h_model
    if h_model == "nn":
        objective = obj_nn
        global activation
        activation = 'relu'
        space = {
            'activation':hp.choice('activation', [('relu', 'relu'), ('tanh', 'tanh'), ('sigmoid','sigmoid')]),
            'n_nodes':hp.uniform('n_nodes', 50, 300),
            'n_feats':hp.uniform('n_feats', 100, 400),
            'anova_name':hp.choice('anova_name', [('f_classif', 'f_classif'), ('chi2', 'chi2')])
        }

    elif h_model == "lstm":
        objective = obj_lstm
        space = {
            'activation':hp.choice('activation', [('relu', 'relu'), ('tanh', 'tanh'), ('sigmoid','sigmoid')]),
            'n_nodes':hp.uniform('n_nodes', 100, 400),
            #'embedding_size':hp.choice('embedding_size', [100, 150, 200, 250, 300, 400]),
            'dropout':hp.uniform('dropout', 0.1, 0.9)
        }
    
    elif h_model == "svm":
        objective = obj_svm
        space = {
            'kernel':hp.choice('kernel', [('rbf', 'rbf'), ('linear', 'linear'), ('poly','poly'), ('sigmoid','sigmoid')]),
            'n_feats':hp.uniform('n_feats', 100, 500),
            'anova_name':hp.choice('anova_name', [('f_classif', 'f_classif'), ('chi2', 'chi2')]),
            'prob':hp.choice('prob', [('True', True), ('False', False)]),
            'weight_classes':hp.choice('weight_classes', [('True', True), ('False', False)])
        }

    elif h_model =="rf":
        objective = obj_rf
        space = {
            'trees':hp.uniform('trees', 3, 30),
            'criterion':hp.choice('criterion', [('gini', 'gini'), ('entropy', 'entropy')]),
            'n_feats':hp.uniform('n_feats', 100, 500),
            'anova_name':hp.choice('anova_name', [('f_classif', 'f_classif'), ('chi2', 'chi2')]),
            'max_feats':hp.uniform('max_feats', 0.0, 1.0),
            'mss':hp.uniform('mss', 1, 5),
            'weight_classes':hp.choice('weight_classes', [('True', True), ('False', False)])
        }

    # Run hyperopt
    print "space: " + str(space)
    best = fmin(objective, space, algo=tpe.suggest, max_evals=100)
    print best

def obj_nn(params):
    activation = params['activation'][0]
    n_nodes = int(params['n_nodes'])
    n_feats = int(params['n_feats'])
    anova_name = params['anova_name'][0]
    print "obj_nn: " + str(activation) + ", nodes:" + str(n_nodes) + ", feats:" + str(n_feats) + ", anova: " + str(anova_name)

    anova_function = None
    if anova_name == 'chi2':
        anova_function = chi2
    elif anova_name == 'f_classif':
        anova_function = f_classif

    # Read in feature keys
    global keys
    with open(h_train + ".keys", "r") as kfile:
        keys = eval(kfile.read())

    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    typeencoder = preprocessing.LabelEncoder()
    trainids = []
    trainlabels = []
    X = []
    Y = []
    Y = preprocess(h_train, trainids, trainlabels, X, Y, True)
    #print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    model, X, Y = create_nn_model(X, Y, anova_function, n_feats, n_nodes, activation)

    # Run test
    testids, testlabels, predictedlabels = test('nn', model, h_test)
    f1score = metrics.f1_score(testlabels, predictedlabels)
    print "F1: " + str(f1score)

    # Return a score to minimize
    return 1 - f1score

def obj_lstm(params):
    activation = params['activation'][0]
    n_nodes = int(params['n_nodes'])
    embedding_size = 200
    dropout = float(params['dropout'])
    print "obj_lstm: " + str(activation) + ", nodes:" + str(n_nodes) + ", embedding_size:" + str(embedding_size) + ", dropout: " + str(dropout)

    # Read in feature keys
    global keys
    with open(h_train + ".keys", "r") as kfile:
        keys = eval(kfile.read())

    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    typeencoder = preprocessing.LabelEncoder()
    trainids = []
    trainlabels = []
    X = []
    Y = []
    Y = preprocess(h_train, trainids, trainlabels, X, Y, True)
    #print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    model, X, Y = create_lstm_model(X, Y, embedding_size, n_nodes, activation, dropout)

    # Run test
    testids, testlabels, predictedlabels = test('lstm', model, h_test)
    f1score = metrics.f1_score(testlabels, predictedlabels)
    print "F1: " + str(f1score)

    # Return a score to minimize
    return 1 - f1score

def obj_svm(params):
    kern = params['kernel'][1]
    n_feats = int(params['n_feats'])
    anova_name = params['anova_name'][1]
    prob = params['prob'][1]
    weight_classes = params['weight_classes'][1]
    print "obj_svm: " + str(kern) + ", prob:" + str(prob) + ", feats:" + str(n_feats) + ", anova: " + str(anova_name)

    anova_function = None
    if anova_name == 'chi2':
        anova_function = chi2
    elif anova_name == 'f_classif':
        anova_function = f_classif

    # Read in feature keys
    global keys
    with open(h_train + ".keys", "r") as kfile:
        keys = eval(kfile.read())

    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    typeencoder = preprocessing.LabelEncoder()
    trainids = []
    trainlabels = []
    X = []
    Y = []
    Y = preprocess(h_train, trainids, trainlabels, X, Y, True)

    global anova_filter
    anova_filter, X = create_anova_filter(X, Y, anova_function, n_feats)
    model = None
    if weight_classes:
        model = svm.SVC(kernel=kern, probability=prob, decision_function_shape='ovr', class_weight='balanced')
    else:
        model = svm.SVC(kernel=kern, probability=prob, decision_function_shape='ovr')
    model.fit(X, Y)

    # Run test
    testids, testlabels, predictedlabels = test('svm', model, h_test)
    f1score = metrics.f1_score(testlabels, predictedlabels)
    print "F1: " + str(f1score)

    # Return a score to minimize
    return 1 - f1score

def obj_rf(params):
    trees = int(params['trees'])
    crit = params['criterion'][1]
    n_feats = int(params['n_feats'])
    anova_name = params['anova_name'][1]
    max_feats = float(params['max_feats'])
    mss = int(params['mss'])
    weight_classes = params['weight_classes'][1]
    print "obj_rf: " + str(trees) + ", crit:" + crit + ", n_feats:" + str(n_feats) + ", anova: " + str(anova_name) + " max_feats: " + str(max_feats) + " mss: " + str(mss) + " weight_classes: " + str(weight_classes)

    anova_function = None
    if anova_name == 'chi2':
        anova_function = chi2
    elif anova_name == 'f_classif':
        anova_function = f_classif

    # Read in feature keys
    global keys
    with open(h_train + ".keys", "r") as kfile:
        keys = eval(kfile.read())

    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    typeencoder = preprocessing.LabelEncoder()
    trainids = []
    trainlabels = []
    X = []
    Y = []
    Y = preprocess(h_train, trainids, trainlabels, X, Y, True)

    global anova_filter
    anova_filter, X = create_anova_filter(X, Y, anova_function, n_feats)
    model = None
    if weight_classes:
        model = RandomForestClassifier(n_estimators=trees, criterion=crit, max_features=max_feats, min_samples_split=mss, class_weight='balanced', n_jobs=-1)
    else:
        model = RandomForestClassifier(n_estimators=trees, criterion=crit, max_features=max_feats, min_samples_split=mss, n_jobs=-1)
    model.fit(X, Y)

    # Run test
    testids, testlabels, predictedlabels = test('rf', model, h_test)
    f1score = metrics.f1_score(testlabels, predictedlabels)
    print "F1: " + str(f1score)

    # Return a score to minimize
    return 1 - f1score


def run(arg_model, arg_modelname, arg_train_feats, arg_test_feats, arg_result_file, arg_prefix, arg_labelname, arg_n_feats=227, arg_anova="chi2", arg_nodes=192, arg_activation='relu', arg_dropout=0.5, arg_rebalance=""):
    total_start_time = time.time()

    # Special handling for neural network models
    is_nn = arg_model == "nn" or arg_model == "lstm" or arg_model == "rnn" or arg_model == "cnn"

    # Params
    num_feats = arg_n_feats
    num_nodes = arg_nodes

    global labelname
    labelname = arg_labelname
    trainids = []        # VA record id
    trainlabels = []     # Correct ICD codes
    X = []               # Feature vectors
    Y = []
    X2 = [] # Extra features for hybrid model
    
    # Read in feature keys
    print "reading feature keys..."
    global keys
    with open(arg_train_feats + ".keys", "r") as kfile:
        keys = eval(kfile.read())
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models

    # If a mix of traditional and vector features is requested, use a hybrid nn model
    vec_keys, point_keys = split_feats(keys)
    hybrid = False
    if len(vec_keys) > 0 and len(point_keys) > 2:
        hybrid = True
        print "hybrid features"

    # Transform ICD codes and record types to numbers
    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder()
    typeencoder = preprocessing.LabelEncoder()

    # Load the features
    if hybrid:
        Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, vec_keys, True)
        preprocess(arg_train_feats, [], [], X2, [], point_keys, True)
    else:
        Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, keys, True)
    print "X: " + str(len(X)) + " Y: " + str(len(Y))
    print "X2: " + str(len(X2))

    # Rebalance
    if arg_rebalance != "":
        print "rebalance: " + arg_rebalance
        X, Y = rebalance.rebalance(X, Y, arg_rebalance)
        print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    # Train the model
    print "training model..."
    stime = time.time()

    # Feature selection
    global anova_filter
    anova_function = f_classif
    if arg_anova == "chi2":
        anova_function = chi2
    print "anova_function: " + arg_anova
    if not is_nn:
        anova_filter, X = create_anova_filter(X, Y, anova_function, num_feats)

    global model
    model = None

    # Neural network models
    if is_nn:
        modelfile = arg_prefix + "/" + arg_modelname + ".model"
        if os.path.exists(modelfile):
            print "using pre-existing model at " + modelfile
            model = load_model(modelfile)
            Y = to_categorical(Y)
            X = numpy.asarray(X)
        else:
            print "creating a new neural network model"
            embedding_dim = 200 
            if arg_model == "nn":
                model, X, Y = create_nn_model(X, Y, anova_function, num_feats, num_nodes, 'relu')
            elif arg_model == "lstm":
                num_nodes = 56
                model, X, Y = create_lstm_model(X, Y, embedding_dim, num_nodes, arg_activation, hybrid=hybrid, X2=X2)
                #score = model.evaluate(X_test, Y_test, batch_size=16
            elif arg_model == "rnn":
                model, X, Y = create_rnn_model(X, Y, embedding_dim, num_nodes, arg_activation)
            elif arg_model == "cnn":
                model, X, Y = create_cnn_model(X, Y, embedding_dim, hybrid=hybrid, X2=X2)

            # Save the model
            print "saving the model..."
            model.save(modelfile)
            plotname = modelfile + ".png"
            plot_model(model, to_file=plotname)

    # Other models
    else:
         if arg_model == "svm":
             print "svm model"
             model = svm.SVC(kernel='linear', decision_function_shape='ovr', probability=True)
         elif arg_model == "knn":
             print "k-nearest neighbor model"
             model = neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=-1)
         elif arg_model == "nb":
             print "naive bayes model"
             model = MultinomialNB()
         elif arg_model == "rf":
             print "random forest model"
             model = RandomForestClassifier(n_estimators=26, max_features=0.0485, min_samples_split=4, class_weight='balanced', n_jobs=-1)

         model.fit(X, Y)

    etime = time.time()
    print "training took " + str(etime - stime) + " s"

    # Test
    testids, testlabels, predictedlabels = test(arg_model, model, arg_test_feats, hybrid)

    # Write results to a file
    output = open(arg_result_file, 'w')
    for i in range(len(testids)):
        out = {}
        out['MG_ID'] = testids[i]
        out['Correct_ICD'] = testlabels[i]
        out['Predicted_ICD'] = predictedlabels[i]
        output.write(str(out) + "\n")
    output.close()

    total_time = (time.time() - total_start_time) / 60
    print "total time: " + str(total_time) + " mins"

def test(model_type, model, testfile, hybrid=False):
    print "testing..."
    stime = time.time()
    testids = []
    testlabels = []
    testX = []
    testX2 = []
    testY = []
    predictedY = []
    if hybrid:
        vec_keys, point_keys = split_feats(keys)
        testY = preprocess(testfile, testids, testlabels, testX, testY, vec_keys)
        preprocess(testfile, [], [], testX2, [], point_keys)
    else:
        testY = preprocess(testfile, testids, testlabels, testX, testY, keys)
    if not model_type == "lstm" and not model_type == "rnn" and not model_type == "cnn":
        testX = anova_filter.transform(testX)
    if model_type == "rnn" or model_type == "lstm" or model_type == "cnn":
        testX = numpy.asarray(testX)

    inputs = [testX]
    if hybrid:
        testX2 = numpy.asarray(testX2)
        inputs.append(testX2)
    print "testX shape: " + str(testX.shape)
    if model_type == "nn" or model_type == "lstm" or model_type == "rnn":
        predictedY = model.predict(inputs)
        results = map_back(predictedY)
    elif model_type == "cnn":
        predictedY = model.predict(inputs)
        results = map_back(predictedY)
        #attn_vec = get_attention_vector(model, testX)
        #print "attention vector: " + str(attn_vec)
    else:
        results = model.predict(testX)

    predictedlabels = labelencoder.inverse_transform(results)
    etime = time.time()
    print "testing took " + str(etime - stime) + " s"
    print "testY: " + str(testY)
    print "results: " + str(results)
    return testids, testlabels, predictedlabels

def create_nn_model(X, Y, anova_function, num_feats, num_nodes, act):
    anova_filter, X = create_anova_filter(X, Y, anova_function, num_feats)
#    X = anova_filter.transform(X)
    Y = to_categorical(Y)

    print "neural network: nodes: " + str(num_nodes) + ", feats: " + str(num_feats)
    nn = Sequential([Dense(num_nodes, input_dim=num_feats),
                    Activation(act),
                    #Dense(num_nodes, input_dim=num_feats),
                    #Activation(activation),
                    Dense(Y.shape[1]),
                    Activation('softmax'),])
        
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(X, Y)
    nn.summary()
    return nn, X, Y

def create_rnn_model(X, Y, embedding_size, num_nodes, act, dropout=0.5):
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)

    print "RNN: nodes: " + str(num_nodes) + " embedding: " + str(embedding_size)
    #print "vocab: " + str(vocab_size)
    print "max_seq_len: " + str(max_seq_len)
    # TEMP for no embedding
    embedding_size = 200
    nn = Sequential([#Embedding(vocab_size, embedding_size, input_length=max_seq_len, mask_zero=False),
                     SimpleRNN(num_nodes, activation=act, return_sequences=False, input_shape=(max_seq_len, embedding_size)),# Dense(200, activation='tanh'),
                     #LSTM(256, input_dim=200, activation='sigmoid', inner_activation='hard_sigmoid'),
                     Dropout(dropout),
                     Dense(Y.shape[1], activation='softmax')])
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    nn.fit(X, Y)
    nn.summary()
    return nn, X, Y

def create_lstm_model(X, Y, embedding_size, num_nodes, activation='sigmoid', dropout=0.1, hybrid=False, X2=[]):
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
    inputs = []
    input_arrays = [X]

    print "LSTM: nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)
    #nn = Sequential([Bidirectional(LSTM(256, input_dim=embedding_size, activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=True), input_shape=(200, embedding_size), merge_mode='concat'),
    #nn = Sequential([LSTM(num_nodes, input_shape=(200, embedding_size), activation='sigmoid', inner_activation='hard_sigmoid', return_sequences=False),
    #                 Dropout(dropout),
                     #Flatten(), # For bidirectional
    #                 Dense(Y.shape[1], activation='softmax')])

    input_shape = (max_seq_len, embedding_size)
    input1 = Input(shape=input_shape)
    inputs.append(input1)
    if hybrid:
        X2 = numpy.asarray(X2)
        print "X2 shape: " + str(X2.shape)
        input_arrays.append(X2)
        input2 = Input(shape=(X2.shape[1],))
        inputs.append(input2)
        ff = Dense(10, activation='relu')(input2)

    lstm_out = LSTM(num_nodes, return_sequences=False)(input1)
    dropout_out = Dropout(dropout)(lstm_out)
    #attn_out = attention(dropout_out, max_seq_len, embedding_size)

    if hybrid:
        #print "ff shape: " + str(ff.output_shape)
        merged = concatenate([dropout_out, ff], axis=-1)

    prediction = Dense(Y.shape[1], activation='softmax')(dropout_out)
    nn = Model(inputs=inputs, outputs=prediction)

    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    nn.fit(input_arrays, Y, nb_epoch=15)
    nn.summary()
    return nn, X, Y

def create_cnn_model(X, Y, embedding_size, act=None, window=3, hybrid=False, X2=[]):
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    embedding_size = X.shape[-1]
    print "train X shape: " + str(X.shape)
    print "CNN: embedding: " + str(embedding_size)
    print "max_seq_len: " + str(max_seq_len)
    print "hybrid: " + str(hybrid)
    window_sizes = [1, 2, 3, 4, 5]
    branches = []
    inputs = []
    input_arrays = [X]

    # Keras functional API with attention
    # Input layers
    input_shape = (max_seq_len, embedding_size) 
    input1 = Input(shape=input_shape)
    inputs.append(input1)
    if hybrid:
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
    for w in window_sizes:
        print "window: " + str(max_seq_len) + " x " + str(w)
        #conv_layer = Conv1D(w, embedding_size, input_shape=input_shape)
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
    nn.fit(input_arrays, Y, epochs=10)
    nn.summary()

    #try:
    #    print "attention weights: " + str(attention_layer.get_weights())
    #except AttributeError:
    #    print "ERROR: got an exception trying to print attention weights"

    return nn, X, Y

def attention(inputs, time_steps, input_dim):
    #input_dim = int(inputs.shape[-1])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, time_steps))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(time_steps, activation='softmax')(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

def get_attention_vector(model, test_input):
    #attention_vectors = []
    #for i in range(300):
    #testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    attention_vector = numpy.mean(attention_utils.get_activations(model, test_input,
                                                   print_shape_only=True,
                                                   layer_name='attention_vec')[0], axis=2).squeeze()
        #print('attention =', attention_vector)
    #assert (numpy.sum(attention_vector) - 1.0) < 1e-5
    #attention_vectors.append(attention_vector)
    #attention_vector_final = np.mean(np.array(attention_vectors), axis=0)
    return attention_vector

def create_anova_filter(X, Y, function, num_feats):
    global anova_filter
    anova_filter = SelectKBest(function, k=num_feats)
    anova_filter.fit(X, Y)
    X = anova_filter.transform(X)
    selected = anova_filter.get_support(True)
    print "features selected: "
    #for i in selected:
    #    print "\t" + keys[i+2]
    return anova_filter, X

'''
    Get the features from the feature file
    filename: the name of the feature file
    ids: the array for the ids
    x: the array for the features
    y: the array for the labels
    feats: a list of the names of the features to extract
    trainlabels: True if this is the trainset, we need to train the label embedding
'''
def preprocess(filename, ids, labels, x, y, feats, trainlabels=False):
    global labelencoder
    ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4"]

    # Read in the feature vectors
    starttime = time.time()
    print "preprocessing features: " + str(feats)
    types = []
    vec_feats = False
    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)
            features = []
            for key in keys:
                if key == 'MG_ID':
                    ids.append(vector[key])
                    #print "ID: " + vector[key]
                elif key == labelname:
                    labels.append(vector[key])
                elif key in feats and key not in ignore_feats: # Only pull out the desired features
                    if key == "CL_type":
                        print "CL_type: " + vector[key]
                        types.append(vector[key])
                    elif key in vec_types:
                        # The feature matrix for word2vec can't have other features
                        features = vector[key]
                        vec_feats = True
                        if key == "narr_seq":
                            global vocab_size
                            vocab_size = vector['vocab_size']
                        global max_seq_len
                        max_seq_len = vector['max_seq_len']
                        #print "max_seq_len: " + str(max_seq_len)
                        features = numpy.asarray(features)#.reshape(max_seq_len, 1)
                        #print "narr_vec shape: " + str(features.shape)
                    elif not vec_feats:
                        if vector.has_key(key):
                            features.append(vector[key])
                        else:
                            features.append('0')
            x.append(features)

    # Convert type features to numerical features
    if len(types) > 0: #and not vec_feats:
        print "converting types to numeric features"
        if trainlabels:
            typeencoder.fit(types)
        enc_types = typeencoder.transform(types)

        # Add the types back to the feature vector
        for i in range(len(x)):
            val = enc_types[i]
            x[i].append(val)
        keys.remove("CL_type")
        keys.append("CL_type")

    # Convert ICD codes to numerical labels
    if trainlabels:
        labelencoder.fit(labels)
    y = labelencoder.transform(labels)

    # Normalize features to 0 to 1 (if not word vectors)
    if not vec_feats:
        preprocessing.minmax_scale(x, copy=False)
    endtime = time.time()
    print "preprocessing took " + str(endtime - starttime) + " s"
    return y

def map_back(results):
    output = []
    for x in range(len(results)):
        res = results[x]
        val = numpy.argmax(res)
        output.append(val)
    return output

def split_feats(keys):
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models
    for key in keys:
        if key in vec_types:
            vec_keys.append(key)
        else:
            point_keys.append(key)
    print "vec_keys: " + str(vec_keys)
    print "point_keys: " + str(point_keys)
    return vec_keys, point_keys

if __name__ == "__main__":main() 
