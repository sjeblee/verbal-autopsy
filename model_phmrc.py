#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import argparse
import numpy
import os
import time
from hyperopt import hp, fmin, tpe, space_eval
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
#from keras.utils.visualize_util import plot
from sklearn import metrics
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

labelencoder = None

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
        print "usage: python model.py --in [train.features] --test [test.features] --out [test.results] --labels [ICD_cat/Final_code] --model [nn] --name [rnn_ngram3] --prefix [/sjeblee/research/models]"
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

        # Define parameter space
        space = {
            'activation':hp.choice('activation', [('relu', 'relu'), ('tanh', 'tanh'), ('sigmoid','sigmoid')]),
            'n_nodes':hp.uniform('n_nodes', 50, 300),
            'n_feats':hp.uniform('n_feats', 100, 400),
            'anova_name':hp.choice('anova_name', [('f_classif', 'f_classif'), ('chi2', 'chi2')])
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


def run(arg_model, arg_modelname, arg_train_feats, arg_test_feats, arg_result_file, arg_prefix, arg_labelname, arg_n_feats=227, arg_anova="chi2", arg_nodes=192):
    total_start_time = time.time()

    # Params
    num_feats = arg_n_feats
    num_nodes = arg_nodes

    global labelname
    labelname = arg_labelname
    is_nn = arg_model == "nn" or arg_model == "lstm"
        
    trainids = []        # VA record id
    trainlabels = []     # Correct ICD codes
    X = []               # Feature vectors
    Y = []
    
    # Read in feature keys
    print "reading feature keys..."
    global keys
    with open(arg_train_feats + ".keys", "r") as kfile:
        keys = eval(kfile.read())
        
    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    typeencoder = preprocessing.LabelEncoder()
    Y = preprocess(arg_train_feats, trainids, trainlabels, X, Y, True)
    print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    # Train model
    print "training model..."
    stime = time.time()

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
            anova_filter, X = create_anova_filter(X, Y, anova_function, num_feats)
            Y = to_categorical(Y)
        else:
            print "creating a new neural network model"
            if arg_model == "nn":
                model, X, Y = create_nn_model(X, Y, anova_function, num_feats, num_nodes, 'tanh')
            elif arg_model == "lstm":
                Y = to_categorical(Y)
                model = Sequential()
                #nn.add(Embedding(max_feat256, input_dim=200))
                model.add(LSTM(256, input_dim=200, activation='sigmoid', inner_activation='hard_sigmoid'))
                model.add(Dropout(0.5))
                model.add(Dense(Y.shape[1]))
                model.add(Activation('sigmoid'))
                model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
                model.fit(numpy.array(X), Y, batch_size=16, nb_epoch=10)
                #score = model.evaluate(X_test, Y_test, batch_size=16)

            # Save the model
            print "saving the model..."
            model.save(modelfile)
            #plotname = modelfile + ".png"
            #plot(nn, to_file=plotname)

    # Other models
    else:
         if arg_model == "svm":
             print "svm model"
             #model = svm.SVC(kernel='rbf', decision_function_shape='ovr')
             model = svm.SVC(kernel='linear', decision_function_shape='ovr', probability=True)
         elif arg_model == "knn":
             print "k-nearest neighbor model"
             model = neighbors.KNeighborsClassifier(n_neighbors=1, weights='distance', n_jobs=-1)
         elif arg_model == "nb":
             print "naive bayes model"
             model = MultinomialNB()
         elif arg_model == "rf":
             print "random forest model"
             model = RandomForestClassifier(n_estimators=27, max_features=0.064, min_samples_split=4, class_weight='balanced', n_jobs=-1)

         model.fit(X, Y)

    etime = time.time()
    print "training took " + str(etime - stime) + " s"

    # Test
    testids, testlabels, predictedlabels = test(arg_model, model, arg_test_feats)

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

def test(model_type, model, testfile):
    print "testing..."
    stime = time.time()
    testids = []
    testlabels = []
    testX = []
    testY = []
    predictedY = []
    testY = preprocess(testfile, testids, testlabels, testX, testY)
    if not model_type == "lstm":
        testX = anova_filter.transform(testX)

    if model_type == "nn" or model_type == "lstm":
        predictedY = model.predict(testX)
        results = map_back(predictedY)
    else:
        results = model.predict(testX)

    predictedlabels = labelencoder.inverse_transform(results)
    etime = time.time()
    print "testing took " + str(etime - stime) + " s"
    print "testY: " + str(testY)
    print "results: " + str(results)
    return testids, testlabels, predictedlabels

def create_nn_model(X, Y, anova_function, num_feats, num_nodes, activation):
    anova_filter, X = create_anova_filter(X, Y, anova_function, num_feats)
#    X = anova_filter.transform(X)
    Y = to_categorical(Y)

    print "neural network: nodes: " + str(num_nodes) + ", feats: " + str(num_feats)
    nn = Sequential([Dense(num_nodes, input_dim=num_feats),
                    Activation(activation),
                    #Dense(num_nodes, input_dim=num_feats),
                    #Activation(activation),
                    Dense(Y.shape[1]),
                    Activation('softmax'),])
        
    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    nn.fit(X, Y)
    nn.summary()
    return nn, X, Y

def create_anova_filter(X, Y, function, num_feats):
    global anova_filter
    anova_filter = SelectKBest(function, k=num_feats)
    anova_filter.fit(X, Y)
    X = anova_filter.transform(X)
    selected = anova_filter.get_support(True)
    print "features selected: "
    for i in selected:
        print "\t" + keys[i+2]
    return anova_filter, X

def preprocess(filename, ids, labels, x, y, trainlabels=False):
    global labelencoder

    # Read in the feature vectors
    starttime = time.time()
    print "preprocessing features..."
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
                elif key == "CL_type":
                    print "CL_type: " + vector[key]
                    types.append(vector[key])
                elif key == "narr_vec":
                    # The feature matrix for word2vec can't have other features
                    features = vector[key]
                    #print "narr_vec shape: " + str(len(features)) + " " + str(len(features[0]))
                    vec_feats = True
                else:
                    if vector.has_key(key):
                        features.append(vector[key])
                    else:
                        features.append('0')
            x.append(features)

    # Convert type features to numerical features
    if len(types) > 0:
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
        #print "res: " + str(res.tolist())
        val = numpy.argmax(res)
        output.append(val)
    return output


if __name__ == "__main__":main() 
