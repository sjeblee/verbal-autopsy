#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import sys
sys.path.append('../keras-attention-mechanism')
sys.path.append('keywords')

import argparse
import numpy
import os
import time
from hyperopt import hp, fmin, tpe, space_eval
from keras.models import Model, Sequential, load_model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Permute, Reshape, RepeatVector, BatchNormalization
from keras.layers import Embedding, LSTM, GRU, TimeDistributed, Merge, merge, concatenate, multiply
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import GlobalMaxPooling1D, MaxPooling1D
from keras.layers.recurrent import SimpleRNN
from keras.layers.wrappers import Bidirectional
from keras.utils.np_utils import to_categorical
from keras.utils import plot_model
from numpy import array, int32, float32
from sklearn import metrics
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

#import attention_utils
import cluster_keywords
import data_util
import model_library
import rebalance
import model_library_torch
#from layers import Attention
from model_dirichlet import create_nn_model

import torch
import pickle
import dill

global anova_filter
labelencoder = None
labelencoder_adult = None
labelencoder_child = None
labelencoder_neonate = None
vec_types = ["narr_vec", "narr_seq", "event_vec", "event_seq", "symp_vec", "kw_vec"]
numpy.set_printoptions(threshold=numpy.inf)

# Use keras or pytorch
use_torch = True

# Output top K features
output_topk_features = True

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
        print "usage: python model.py --in [train.features] --test [test.features] --out [test.results] --labels [ICD_cat/ICD_cat_neo/Final_code] --model [nn/cnn/lstm/gru/filternn] --name [rnn_ngram3] --prefix [/sjeblee/research/models]"
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
            'anova_name':hp.choice('anova_name', [('f_classif', 'f_classif'), ('chi2', 'chi2')]),
	    'threshold':hp.choice('threshold',[0.0,0.05,0.1,0.15,0.2,0.25,3]),
	    'num_epochs':hp.choice('num_epochs', [10,15,20,25,30])
        }

    if h_model == "cnn":
	objective = obj_cnn
	global activation
	activation = 'relu'
	space = {
	    #'activation':hp.choice('activation', [('relu','relu'), ('tanh','tank'),('sigmoid','sigmoid')]),
	    'dropout':hp.uniform('dropout',0,0.01),
	    'threshold':hp.choice('threshold',[0.0,0.05,0.1,0.15]),
	    'kernel_sizes':hp.choice('kernel_sizes', [5,6,7,8]),
	    'num_epochs':hp.choice('num_epochs', [10,15,20,25,30])
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
    threshold = float(params['threshold'])
    num_epochs = int(params['num_epochs'])
    print "obj_nn: " + str(activation) + ", nodes:" + str(n_nodes) + ", feats:" + str(n_feats) + ", anova: " + str(anova_name) + ", threshold: " + str(threshold) + ", num_epochs:" + str(num_epochs)

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
    Y = preprocess(h_train, trainids, trainlabels, X, Y, keys, trainlabels=True)
    #print "X: " + str(len(X)) + "\nY: " + str(len(Y))
    if use_torch == False:
	model, X, Y = create_nn_model(X, Y, anova_function, n_feats, n_nodes, activation)
    else: # Use pytorch model
	Y = to_categorical(Y)
	model = model_library_torch.nn_model(X,Y,n_nodes,activation, num_epochs=num_epochs)
    # Run test
    testids, testlabels, predictedlabels = test('nn', model, h_test,threshold=threshold)
    print "Real Labels shape: " + str(testlabels)
    print "Predicted Labels shape: " + str(predictedlabels)
    f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
    print "F1: " + str(f1score)

    # Return a score to minimize
    return 1 - f1score

def obj_cnn(params):
    dropout = float(params['dropout'])
    threshold = float(params['threshold'])
    kernel_sizes = int(params['kernel_sizes'])
    num_epochs = int(params['num_epochs'])
    print "obj_cnn: " + "dropout = " + str(dropout) + ", threshold = " + str(threshold) + ", kernel_sizes = " + str(kernel_sizes) + ", num_epochs = " + str(num_epochs)

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
    Y = preprocess(h_train, trainids, trainlabels, X, Y, keys, trainlabels=True)

    Y = to_categorical(Y)
    if use_torch == False:
	model, X, Y = model_library.cnn_model(X,Y)
    else:
	model = model_library_torch.cnn_model(X,Y, dropout=dropout, kernel_sizes=kernel_sizes, num_epochs=num_epochs)

    testids, testlabels, predictedlabels = test('cnn', model, h_test, threshold=threshold)
    print "Real Labels shape: " + str(testlabels)
    print "Predicted Labels shape: " + str(predictedlabels)
    f1score = metrics.f1_score(testlabels, predictedlabels, average='weighted')
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


def run(arg_model, arg_modelname, arg_train_feats, arg_test_feats, arg_result_file, arg_prefix, arg_labelname, arg_n_feats=500, arg_anova="chi2", arg_nodes=192, arg_activation='relu', arg_dropout=0.5, arg_rebalance=""):
    total_start_time = time.time()

    # Special handling for neural network models
    is_nn = ("nn" in arg_model) or 'gru' in arg_model or 'lstm' in arg_model

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

    # Figure out record type
    rec_type = None
    if 'adult' in arg_result_file:
        rec_type = 'adult'
    elif 'child' in arg_result_file:
        rec_type = 'child'
    elif 'neo' in arg_result_file:
        rec_type = 'neonate'
    
    # Read in feature keys
    print "reading feature keys..."
    global keys
    with open(arg_train_feats + ".keys", "r") as kfile:
        keys = eval(kfile.read())
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models

    # If a mix of traditional and vector features is requested, use a hybrid nn model
    vec_keys, point_keys = split_feats(keys, arg_labelname)
    hybrid = False
    if len(vec_keys) > 0 and len(point_keys) > 2:
        hybrid = True
        print "hybrid features"

    # Transform ICD codes and record types to numbers
    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder()                        
    typeencoder = preprocessing.LabelEncoder()

    # TEMP
    #if "keyword_clusters" in keys:
    #    hybrid = False

    # Load the features
    if hybrid:
        Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, vec_keys, trainlabels=True)
        preprocess(arg_train_feats, [], [], X2, [], point_keys, trainlabels=True)
    else:
	print("Keys tested")
        Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, keys, trainlabels=True, Y2_labels='keyword_clusters')
    print "X: " + str(len(X)) + " Y: " + str(len(Y))
    print "X2: " + str(len(X2))
    x_feats = numpy.asarray(X).shape[-1]

    # TEMP
    #outfile = open('filternn_ids', 'w')
    #outfile.write(str(trainids))
    #outfile.close()

    # Rebalance
    if arg_rebalance != "":
        print "rebalance: " + arg_rebalance
        X, Y = rebalance.rebalance(X, Y, arg_rebalance)
        print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    # Train the model
    print "training model..."
    stime = time.time()

    # Feature selection
    #global anova_filter
    anova_filter = None
    anova_function = f_classif
    if arg_anova == "chi2":
        anova_function = chi2
    print "anova_function: " + arg_anova
    if ((not is_nn) or arg_model == "nn") and num_feats < x_feats:
        anova_filter, X = create_anova_filter(X, Y, anova_function, num_feats)

    # Select k best features for each class
    if output_topk_features == True:
        if arg_model == "nn":
	    select_top_k_features_per_class(X,Y,anova_function,arg_prefix, 100)

    global model
    model = None
    cnn_model = None

    # Neural network models
    if is_nn:
        modelfile = arg_prefix + "/" + arg_modelname + ".model"
        if os.path.exists(modelfile):
            print "using pre-existing model at " + modelfile

	    if use_torch:
		model = torch.load(modelfile)
	    else:
		model = load_model(modelfile)
            X = numpy.asarray(X)

	# For CNN-RNN ensemble
	elif arg_model == "cnn-rnn":
	    cnn_modelfile = arg_prefix + "/" + arg_modelname + "_cnn.model"
	    rnn_modelfile = arg_prefix + "/" + arg_modelname + "_rnn.model"
	    if (os.path.exists(cnn_modelfile) and os.path.exists(rnn_modelfile)):
		print "using pre-existing model at " + cnn_modelfile + " and " + rnn_modelfile
		cnn_input_model = torch.load(cnn_modelfile)
		rnn_model = torch.load(rnn_modelfile)
		model = [cnn_input_model, rnn_model]
		X = numpy.asarray(X)
	    else:
		print "creating a new cnn-rnn ensemble model"
		if use_torch:
                    Y = to_categorical(Y)
                    cnn_input_model, rnn_model = model_library_torch.cnn_attnrnn(X,Y)
                    model = [cnn_input_model, rnn_model]

		    # Save both models
		    torch.save(cnn_input_model,cnn_modelfile)
                    torch.save(rnn_model, rnn_modelfile)
                else:
                    print "cnn-rnn model must use pytorch"
        else:
            print "creating a new neural network model"
            embedding_dim = 100
            #if "keyword_clusters" in keys:
            #    num_nodes = 100
            #    model, cnn_model, X, Y = model_library.rnn_keyword_model(X, Y, num_nodes, arg_activation, arg_model, keywords=X2, num_epochs=15)
            if '_' in arg_model:
                num_nodes = 100
                Y_arrays = [to_categorical(Y[0]), numpy.asarray(Y[1])]
                #Y_arrays = [numpy.asarray(Y[1])]
                model, X, Y = model_library.stacked_model(X, Y_arrays, num_nodes, arg_activation, models=arg_model, loss_func='mean_squared_error')
            elif arg_model == "nn":
                if len(X) == 0 and len(X2) > 0:
                    X = X2
                Y = to_categorical(Y)
		if use_torch:
		    model = model_library_torch.nn_model(X,Y,num_nodes,'relu')
		else:
		    model, X, Y = model_library.nn_model(X, Y, num_nodes, 'relu')
            elif arg_model == "rnn" or arg_model == "lstm" or arg_model == "gru":
                num_nodes = 100
                #Y = to_categorical(Y)
                model, X, Y = model_library.rnn_model(X, Y, num_nodes, arg_activation, arg_model, X2=X2)
            elif arg_model == "rnn_cnn":
                Y = to_categorical(Y)
                num_nodes = 100
                model, X, Y = model_library.rnn_cnn_model(X, Y, num_nodes, arg_activation, arg_model, X2=X2)
            elif arg_model == "cnn":
                Y = to_categorical(Y)
		if use_torch:
		    model = model_library_torch.cnn_model(X,Y)
		else:
                    model, X, Y = model_library.cnn_model(X, Y, X2=X2)
            elif arg_model == "filternn":
                num_nodes = 56
                Y = to_categorical(Y)
                model, X, Y = create_filter_rnn_model(X, Y, embedding_dim, num_nodes)
	  
	    print "saving the model..."

	    if not use_torch: # Save Keras model
	        model.save(modelfile)
	        plotname = modelfile + ".png"
	        plot_model(model, to_file=plotname)
	    else: # Save Pytorch model
	        torch.save(model, modelfile)
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
    if '_' in arg_model:
        testids, testlabels, predictedlabels = test_multi(arg_model, model, arg_test_feats)
    else:
        testids, testlabels, predictedlabels = test(arg_model, model, arg_test_feats, anova_filter, hybrid, kw_cnn=cnn_model)

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

def run_joint(arg_model, arg_modelname, arg_train_feats, arg_test_feats, arg_results_file, arg_prefix, arg_labelname, arg_n_feats=227, arg_anova="chi2", arg_nodes=192, arg_activation='relu', arg_dropout=0.5, arg_rebalance=""):
    total_start_time = time.time()

    # Special handling for neural network models
    is_nn = ("nn" in arg_model) or arg_model == 'gru' or arg_model == 'lstm'

    # Params
    num_feats = arg_n_feats
    num_nodes = arg_nodes

    global labelname
    labelname = arg_labelname
    trainids_adult = []        # VA record id
    trainids_child = []
    trainids_neonate = []
    trainlabels_adult = []     # Correct ICD codes
    trainlabels_child = []
    trainlabels_neonate = []
    X_adult = []               # Feature vectors
    X_child = []
    X_neonate = []
    Y_adult = []
    Y_child = []
    Y_neonate = []
    X2_adult = [] # Extra features for hybrid model
    X2_child = []
    X2_neonate = []
    
    # Read in feature keys
    print "reading feature keys..."
    global keys
    with open(arg_train_feats[0] + ".keys", "r") as kfile:
        keys = eval(kfile.read())
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models

    # If a mix of traditional and vector features is requested, use a hybrid nn model
    vec_keys, point_keys = split_feats(keys, arg_labelname)
    hybrid = False
    if len(vec_keys) > 0 and len(point_keys) > 2:
        hybrid = True
        print "hybrid features"

    # Transform ICD codes and record types to numbers
    global labelencoder, labelencoder_adult, labelencoder_child, labelencoder_neonate
    labelencoder = None
    labelencoder_adult = preprocessing.LabelEncoder()
    labelencoder_child = preprocessing.LabelEncoder()
    labelencoder_neonate = preprocessing.LabelEncoder()

    # Load the features
    if hybrid:
        keys = vec_keys
        #print "hybrid not supported for joint model!"
        #Y_adult = preprocess(arg_train_feats[0], trainids, trainlabels, X_adult, Y_adult, vec_keys, True)
        #preprocess(arg_train_feats, [], [], X2, [], point_keys, True)
    Y_adult = preprocess(arg_train_feats[0], trainids_adult, trainlabels_adult, X_adult, Y_adult, keys, rec_type='adult', trainlabels=True)
    Y_child = preprocess(arg_train_feats[1], trainids_child, trainlabels_child, X_child, Y_child, keys, rec_type='child', trainlabels=True)
    Y_neonate = preprocess(arg_train_feats[2], trainids_neonate, trainlabels_neonate, X_neonate, Y_neonate, keys, rec_type='neonate', trainlabels=True)
    if hybrid:
        preprocess(arg_train_feats[0], [], [], X2_adult, [], point_keys, rec_type='adult', trainlabels=True)
        preprocess(arg_train_feats[1], [], [], X2_child, [], point_keys, rec_type='child', trainlabels=True)
        preprocess(arg_train_feats[2], [], [], X2_neonate, [], point_keys, rec_type='neonate', trainlabels=True)
        
    print "adult X: " + str(len(X_adult)) + " Y: " + str(len(Y_adult))
    print "child X: " + str(len(X_child)) + " Y: " + str(len(Y_child))
    print "neonate X: " + str(len(X_neonate)) + " Y: " + str(len(Y_neonate))
    #Y_adult = to_categorical(Y_adult)
    #Y_child = to_categorical(Y_child)
    #Y_neonate = to_categorical(Y_neonate)
    #print "X2: " + str(len(X2))
    #outfile = open('filternn_ids', 'w')
    #outfile.write(str(trainids))
    #outfile.close()

    # Rebalance
    #if arg_rebalance != "":
    #    print "rebalance: " + arg_rebalance
    #    X, Y = rebalance.rebalance(X, Y, arg_rebalance)
    #    print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    # Train the model
    print "training model..."
    stime = time.time()

    global model
    model = None

    # Neural network models
    modelfile = arg_prefix + "/" + arg_modelname + ".model"
    #if os.path.exists(modelfile):
    #    print "using pre-existing model at " + modelfile
    #    model = load_model(modelfile)
    #    Y = to_categorical(Y)
    #    X = numpy.asarray(X)
    #else:
    print "creating a new neural network model"
    embedding_dim = 200 
        #if arg_model == "nn":
        #    model, X, Y = create_nn_model(X, Y, anova_function, num_feats, num_nodes, 'relu')
    if arg_model == "lstm" or arg_model == "gru" or arg_model == "rnn" or arg_model=="cnn":
        num_nodes = 100

        # ADULT
        print "ADULT"
        X_pretrain = [X_child, X_neonate]
        Y_pretrain = [Y_child, Y_neonate]
        X2_pretrain = []
        if hybrid:
            X2_pretrain = [X2_child, X2_neonate]
        if arg_model == "cnn":
            model_adult, X_adult_out, Y_adult_out = model_library.cnn_model(X_adult, to_categorical(Y_adult), X2=X2_adult, pretrainX=X_pretrain, pretrainY=Y_pretrain, pretrainX2=X2_pretrain, num_epochs=10)
        else:
            model_adult, X_adult_out, Y_adult_out = model_library.rnn_model(X_adult, Y_adult, num_nodes, modelname=arg_model, X2=X2_adult, pretrainX=X_pretrain, pretrainY=Y_pretrain, pretrainX2=X2_pretrain, num_epochs=10)
        testids_adult, testlabels_adult, predictedlabels_adult = test(arg_model, model_adult, arg_test_feats[0], hybrid=hybrid, rec_type='adult')

        # CHILD
        print "CHILD"
        X_pretrain = [X_adult, X_neonate]
        Y_pretrain = [Y_adult, Y_neonate]
        X2_pretrain = []
        if hybrid:
            X2_pretrain = [X2_adult, X2_neonate]
        if arg_model == "cnn":
             model_child, X_child_out, Y_child_out = model_library.cnn_model(X_child, to_categorical(Y_child), X2=X2_child, pretrainX=X_pretrain, pretrainY=Y_pretrain, pretrainX2=X2_pretrain, num_epochs=15)
        else:
             model_child, X_child_out, Y_child_out = model_library.rnn_model(X_child, Y_child, num_nodes, modelname=arg_model, X2=X2_child, pretrainX=X_pretrain, pretrainY=Y_pretrain, pretrainX2=X2_pretrain, num_epochs=15)
        testids_child, testlabels_child, predictedlabels_child = test(arg_model, model_child, arg_test_feats[1], hybrid=hybrid, rec_type='child')
        
        # NEONATE
        print "NEONATE"
        X_pretrain = [X_adult, X_child]
        Y_pretrain = [Y_adult, Y_child]
        X2_pretrain = []
        if hybrid:
            X2_pretrain = [X2_adult, X2_child]
        if arg_model == "cnn":
            model_neonate, X_neonate_out, Y_neonate_out = model_library.cnn_model(X_neonate, to_categorical(Y_neonate), X2=X2_neonate, pretrainX=X_pretrain, pretrainY=Y_pretrain, pretrainX2=X2_pretrain, num_epochs=15)
        else:
            model_neonate, X_neonate_out, Y_neonate_out = model_library.rnn_model(X_neonate, Y_neonate, num_nodes, modelname=arg_model, X2=X2_neonate, pretrainX=X_pretrain, pretrainY=Y_pretrain, pretrainX2=X2_pretrain, num_epochs=15)
        testids_neonate, testlabels_neonate, predictedlabels_neonate = test(arg_model, model_neonate, arg_test_feats[2], hybrid=hybrid, rec_type='neonate')

        #elif arg_model == "cnn":
        #    model, X, Y = create_cnn_model(X, Y, embedding_dim, hybrid=hybrid, X2=X2)
        #elif arg_model == "filternn":
        #    num_nodes = 56
        #    model, X, Y = create_filter_rnn_model(X, Y, embedding_dim, num_nodes)

        # Save the model
        #print "saving the model..."
        #model.save(modelfile)
        #plotname = modelfile + ".png"
        #plot_model(model, to_file=plotname)

    # Other models
    else:
         print "ERROR: joint does not support model " + arg_model

    etime = time.time()
    print "training took " + str(etime - stime) + " s"

    # Write results to file
    write_results(arg_results_file[0], testids_adult, testlabels_adult, predictedlabels_adult)
    write_results(arg_results_file[1], testids_child, testlabels_child, predictedlabels_child)
    write_results(arg_results_file[2], testids_neonate, testlabels_neonate, predictedlabels_neonate)

    total_time = (time.time() - total_start_time) / 60
    print "total time: " + str(total_time) + " mins"

def write_results(filename, testids, testlabels, predictedlabels):

    # Write results to a file
    output = open(filename, 'w')
    for i in range(len(testids)):
        out = {}
        out['MG_ID'] = testids[i]
        out['Correct_ICD'] = testlabels[i]
        out['Predicted_ICD'] = predictedlabels[i]
        output.write(str(out) + "\n")
    output.close()

def test(model_type, model, testfile, anova_filter=None, hybrid=False, rec_type=None, kw_cnn=None, threshold=0.01):
    print "testing..."
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
        print "testX shape: " + str(testX.shape)

    inputs = [testX]
    if hybrid:
        testX2 = numpy.asarray(testX2)
        #if "keyword_clusters" not in keys:
        inputs.append(testX2)
    if is_nn:
	if use_torch: # Test using pytorch model
	    if model_type == "nn": 
		results = model_library_torch.test_nn(model,testX,threshold=threshold)
	    elif model_type == "cnn":
		results = model_library_torch.test_cnn(model,testX, testY,threshold=threshold)
	    elif model_type == "cnn-rnn":
		results = model_library_torch.test_cnn_attnrnn(model[0],model[1], testX, testY)
	else: # Test using Keras model
	    predictedY = model.predict(inputs)
	    results = map_back(predictedY)
	print "testX shape: " + str(testX.shape)
    #elif model_type == "cnn":
    #    attn_vec = get_attention_vector(model, testX)
    #    print "attention vector: " + str(attn_vec)
    else:
        results = model.predict(testX)

    # Score keywords from keyword model
    if kw_cnn is not None:
        kw_pred = kw_cnn.predict(testX)
        kw_pred = data_util.map_to_multi_hot(kw_pred)
        testX2 = testX2.tolist()
        #print "testx[0]: " + str(testx[0])
        #print "testy[0]: " + str(type(testy[0])) + " " + str(testy[0])
        #print "testy_pred[0]: " + str(type(testy_pred[0])) + " " + str(testy_pred[0])

        # Decode labels
        kw_pred_labels = data_util.decode_multi_hot(kw_pred)
        print "kw_pred_labels[0]: " + str(kw_pred_labels[0])
        clusterfile = "/u/sjeblee/research/va/data/datasets/mds+rct/train_adult_cat_spell.clusters"
        kw_pred_text = cluster_keywords.interpret_clusters(kw_pred_labels, clusterfile)
        kw_true_text = cluster_keywords.interpret_clusters(data_util.decode_multi_hot(testX2), clusterfile)
        print "kw_pred_text[0]: " + str(kw_pred_text[0])
        print "kw_true_text[0]: " + str(kw_true_text[0])

        # Score results against nearest neighbor classifier
        print "Scores for 1 class (0.1 cutoff):"
        precision, recall, f1, micro_p, micro_r, micro_f1 = data_util.score_vec_labels(testX2, kw_pred)
        print "Macro KW scores:"
        print "F1: " + str(f1)
        print "precision: " + str(precision)
        print "recall: " + str(recall)
        print "Micro KW scores:"
        print "F1: " + str(micro_f1)
        print "precision: " + str(micro_p)
        print "recall: " + str(micro_r)

    labenc = labelencoder
    if rec_type == 'adult':
        labenc = labelencoder_adult
    elif rec_type == 'child':
        labenc = labelencoder_child
    elif rec_type == 'neonate':
        labenc = labelencoder_neonate

    # Print out classes for index location of each class in the list
    print "Index location of each class: "
    print(str(labenc.classes_))

    predictedlabels = labenc.inverse_transform(results)
    etime = time.time()
    print "testing took " + str(etime - stime) + " s"
    print "testY: " + str(testY)
    print "results: " + str(results)
    print "predicted labels: " + str(predictedlabels)
    return testids, testlabels, predictedlabels

def test_multi(model_type, model, testfile, rec_type=None):
    print "testing multi..."
    stime = time.time()
    testids = []
    testlabels = []
    testX = []
    testY2 = []
    testY = []
    predictedY = []
    #if hybrid:
    #    vec_keys, point_keys = split_feats(keys, labelname)
    #    testY = preprocess(testfile, testids, testlabels, testX, testY, vec_keys)
    #    preprocess(testfile, [], [], testX2, [], point_keys)
    #else:
    Y_arrays = preprocess(testfile, testids, testlabels, testX, testY, keys, rec_type, Y2_labels='keyword_clusters')
    print "Y_arrays: " + str(len(Y_arrays))
    testY = Y_arrays[0]
    testY2 = Y_arrays[1]
    testX = numpy.asarray(testX)
    print "testX shape: " + str(testX.shape)
    print "testY2 shape: " + str(len(testY2))
    #print "testY2: " + str(type(testY2)) + " : " + str(testY2)

    inputs = [testX]
    #if hybrid:
    #    testX2 = numpy.asarray(testX2)
    #    #if "keyword_clusters" not in keys:
    #    inputs.append(testX2)

    predictedY = model.predict(inputs)
    results = map_back(predictedY[0])
    kw_pred = predictedY[1]
    #kw_pred = model.predict(inputs)
    print "kw_pred: " + str(len(kw_pred))

    # Score keywords from keyword model
    kw_pred = data_util.map_to_multi_hot(kw_pred)
    #testX2 = testX2.tolist()
    #print "testx[0]: " + str(testx[0])
    #print "testy[0]: " + str(type(testy[0])) + " " + str(testy[0])
    #print "testy_pred[0]: " + str(type(testy_pred[0])) + " " + str(testy_pred[0])

    # Decode labels
    kw_pred_labels = data_util.decode_multi_hot(kw_pred)
    print "kw_pred_labels[0]: " + str(kw_pred_labels[0])
    #clusterfile = "/u/sjeblee/research/va/data/datasets/mds+rct/train_adult_cat_spell.clusters_km2"
    #kw_pred_text = cluster_keywords.interpret_clusters(kw_pred_labels, clusterfile)
    #kw_true_text = cluster_keywords.interpret_clusters(data_util.decode_multi_hot(testY2), clusterfile)
    #print "kw_pred_text[0]: " + str(kw_pred_text[0])
    #print "kw_true_text[0]: " + str(kw_true_text[0])

    # Score results against nearest neighbor classifier
    print "Scores for 1 class (0.1 cutoff):"
    precision, recall, f1, micro_p, micro_r, micro_f1 = data_util.score_vec_labels(testY2, kw_pred)
    print "Macro KW scores:"
    print "F1: " + str(f1)
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "Micro KW scores:"
    print "F1: " + str(micro_f1)
    print "precision: " + str(micro_p)
    print "recall: " + str(micro_r)

    labenc = labelencoder
    if rec_type == 'adult':
        labenc = labelencoder_adult
    elif rec_type == 'child':
        labenc = labelencoder_child
    elif rec_type == 'neonate':
        labenc = labelencoder_neonate
    predictedlabels = labenc.inverse_transform(results)
    etime = time.time()
    print "testing took " + str(etime - stime) + " s"
    print "testY: " + str(testY)
    print "results: " + str(results)
    return testids, testlabels, predictedlabels

def train_test_joint_lstm_model(X_adult, Y_adult, X_child, Y_child, X_neo, Y_neo, arg_test_feats, arg_results_file, embedding_size, num_nodes, activation='sigmoid', modelname='lstm', dropout=0.1, hybrid=False, X2=[]):
    Y_adult = to_categorical(Y_adult)
    X_adult = numpy.asarray(X_adult)
    print "train X shape adult: " + str(X_adult.shape)
    Y_child = to_categorical(Y_child)
    X_child = numpy.asarray(X_child)
    print "train X shape child: " + str(X_child.shape)
    Y_neo = to_categorical(Y_neo)
    X_neo = numpy.asarray(X_neo)
    print "train X shape neo: " + str(X_neo.shape)
    embedding_size = X_adult.shape[-1]
    inputs = []
    #input_arrays = [X]

    print "model: " + modelname + " nodes: " + str(num_nodes) + " embedding: " + str(embedding_size) + " max_seq_len: " + str(max_seq_len)

    rnn_states = None
    input_shape = (max_seq_len, embedding_size)
    input1 = Input(shape=input_shape)
    inputs.append(input1)
    # TODO: fix hybrid to work with joint training
    if hybrid:
        X2 = numpy.asarray(X2)
        print "X2 shape: " + str(X2.shape)
        inputs.append(X2)
        input2 = Input(shape=(X2.shape[1],))
        inputs.append(input2)
        ff = Dense(10, activation='relu')(input2)

    if modelname == 'gru':
        rnn = GRU(num_nodes, return_sequences=False, return_state=True)
    else:
        rnn = LSTM(num_nodes, return_sequences=False, return_state=True)
    if rnn_states == None:
        rnn_out, rnn_states = rnn(input1)
    else:
        rnn_out, rnn_states = rnn(input1, initial_state=rnn_states)
    dropout_out = Dropout(dropout)(rnn_out)
    #attn_out = attention(dropout_out, max_seq_len, embedding_size)

    if hybrid:
        #print "ff shape: " + str(ff.output_shape)
        merged = concatenate([dropout_out, ff], axis=-1)

    prediction_adult = Dense(Y_adult.shape[1], activation='softmax')(dropout_out)
    prediction_child = Dense(Y_child.shape[1], activation='softmax')(dropout_out)
    prediction_neo = Dense(Y_neo.shape[1], activation='softmax')(dropout_out)
    
    model_adult = Model(inputs=inputs, outputs=prediction_adult)
    model_adult.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model_child = Model(inputs=inputs, outputs=prediction_child)
    model_child.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model_neo = Model(inputs=inputs, outputs=prediction_neo)
    model_neo.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model_adult.fit(X_adult, Y_adult, epochs=15)
    model_adult.summary()
    print "test feats: " + arg_test_feats[0]
    testids_adult, testlabels_adult, predictedlabels_adult = test(modelname, model_adult, arg_test_feats[0], hybrid, 'adult')
    write_results(arg_results_file[0], testids_adult, testlabels_adult, predictedlabels_adult)
    
    model_child.fit(X_child, Y_child, epochs=15)
    model_child.summary()
    print "test feats: " + arg_test_feats[1]
    testids_child, testlabels_child, predictedlabels_child = test(modelname, model_child, arg_test_feats[1], hybrid, 'child')
    write_results(arg_results_file[1], testids_child, testlabels_child, predictedlabels_child)
    
    model_neo.fit(X_neo, Y_neo, epochs=15)
    model_neo.summary()
    print "test feats: " + arg_test_feats[2]
    testids_neonate, testlabels_neonate, predictedlabels_neonate = test(modelname, model_neo, arg_test_feats[2], hybrid, 'neonate')
    write_results(arg_results_file[2], testids_neonate, testlabels_neonate, predictedlabels_neonate)

    return model_adult, model_child, model_neo

def create_filter_rnn_model(X, Y, embedding_size, num_nodes, activation='tanh', dropout=0.1, hybrid=False, X2=[]):
    Y = to_categorical(Y)
    X = numpy.asarray(X)
    print "train X shape: " + str(X.shape)
    embedding_size = X.shape[-1]
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

#def get_attention_vector(model, test_input):
    #attention_vectors = []
    #for i in range(300):
    #testing_inputs_1, testing_outputs = get_data_recurrent(1, TIME_STEPS, INPUT_DIM)
    #attention_vector = numpy.mean(attention_utils.get_activations(model, test_input,
    #                                               print_shape_only=True,
    #                                               layer_name='attention_vec')[0], axis=2).squeeze()
        #print('attention =', attention_vector)
    #assert (numpy.sum(attention_vector) - 1.0) < 1e-5
    #attention_vectors.append(attention_vector)
    #attention_vector_final = np.mean(np.array(attention_vectors), axis=0)

    #return attention_vector

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
def preprocess(filename, ids, labels, x, y, feats, rec_type=None, trainlabels=False, Y2_labels=None):
    global labelencoder, labelencoder_adult, labelencoder_child, labelencoder_neonate
    #ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4","symp_vec"]
    ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4"]

    # Read in the feature vectors
    starttime = time.time()
    print "preprocessing features: " + str(feats)
    types = []
    kw_clusters = []
    vec_feats = False
    Y_arrays = []
    labels2 = []
    extra_labels = False

    # Edit by Yoona for narrative symptoms
    #symptoms_keys = []
    #symptoms_keys_fixed = False

    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)
            features = []

            # Edit by Yoona. Fix keys in narrative symptoms vector. 
            #if not symptoms_keys_fixed:
            #    if "narr_symptoms" in keys:
            #        symptoms_vec = vector["narr_symptoms"]
            #        symptoms_keys = symptoms_vec.keys()
            #        symptoms_keys_fixed = True

            for key in keys:
            #for key in feats:
                if key == 'MG_ID':
                    ids.append(vector[key])
                    #print "ID: " + vector[key]
                elif key == labelname:
                    labels.append(vector[key])
                elif key in feats and key not in ignore_feats: # Only pull out the desired features
                    if key == "CL_type":
                        print "CL_type: " + vector[key]
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
			    #print "This is: " + str(key) + "and value is : " + str(vector[key])
                            features.append(vector[key])
                        else:
                            features.append('0')

                # Edit by Yoona for appending narrative symptoms
                #elif key == "narr_symptoms":
                #    symp_vector = vector[key]
                #    for symp_key in symptoms_keys:
                #        if symp_vector.has_key(symp_key):
                #            features.append(symp_vector[symp_key])
                #        else:
                #            features.append('0')
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

    # Convert the keyword clusters to multi-hot vectors
    if len(kw_clusters) > 0:
        use_multi_hot = True
        #use_multi_hot = (Y2_labels == 'keyword_clusters')
        if use_multi_hot:
            print "converting keyword clusters to multi-hot vectors"
            cluster_feats = data_util.multi_hot_encoding(kw_clusters, 100)
            labels2 = cluster_feats
            #keys.remove("keyword_clusters")
            #extra_labels = True

        else:
            # Convert keywords to cluster embeddings
            print "converting clusters to embeddings..."
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

    # Convert ICD codes to numerical labels
    labenc = labelencoder
    if rec_type == 'adult':
        print "using adult labelencoder"
        labenc = labelencoder_adult
    elif rec_type == 'child':
        print "using child labelencoder"
        labenc = labelencoder_child
    elif rec_type == 'neonate':
        print "using neonate labelencoder"
        labenc = labelencoder_neonate
    if trainlabels:
        labenc.fit(labels)
    print(labels)
    y = labenc.transform(labels)

    # Normalize features to 0 to 1 (if not word vectors)
    if not vec_feats:
	preprocessing.minmax_scale(x, copy=False)
    endtime = time.time()
    mins = float(endtime-starttime)/60
    print "preprocessing took " + str(mins) + " mins"
    if extra_labels:
        print "returning extra labels"
        return [y, labels2]
    else:
        return y

def map_back(results):
    output = []
    for x in range(len(results)):
        res = results[x]
        val = numpy.argmax(res)
        output.append(val)
    return output

def split_feats(keys, labelname):
    ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4"] # symp_vec added by Yoona
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models
    for key in keys:
        if key in vec_types:
            vec_keys.append(key)
        elif key == labelname or key not in ignore_feats:
            point_keys.append(key)

    print "vec_keys: " + str(vec_keys)
    print "point_keys: " + str(point_keys)
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
	print "Class: " + str(classes[i])
	this_Y = []
	for j in range(len(Y)):
	    if Y[j] == i:
		binary = 1
	    else:
		binary = 0
	    this_Y.append(binary)
	anova_symp = SelectKBest(function, k)
	anova_symp.fit(X,this_Y)
	best_indices = anova_symp.get_support(True)
	print "Best indices: " + str(best_indices)
	for i in range(len(best_indices)):
	    selected = str(keys[best_indices[i] + 2])
	    print selected
	    output.write("\n")
	    output.write(selected)

if __name__ == "__main__":main() 
