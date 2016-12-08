#!/usr/bin/python
# Build an Neural Network classifier with the VA features
# @author sjeblee@cs.toronto.edu

import argparse
import numpy
import os
import time
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
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

    if not (args.infile and args.outfile and args.testfile):
        print "usage: python svm.py --in [train.features] --test [test.features] --out [test.results] --labels [ICD_cat/Final_code] --model [nn] --name [rnn_ngram3] --prefix [/sjeblee/research/models]"
        exit()

    # Params
    num_feats = 200

    global labelname
    labelname = "Final_code"
    if args.labelname:
        labelname = args.labelname
    model = "nn"
    if args.model:
        model = args.model
        
    trainids = []        # VA record id
    trainlabels = []     # Correct ICD codes
    X = []               # Feature vectors
    Y = []
    
    # Read in feature keys
    global keys
    with open(args.infile + ".keys", "r") as kfile:
        keys = eval(kfile.read())
        
    global labelencoder, typeencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    typeencoder = preprocessing.LabelEncoder()
    Y = preprocess(args.infile, trainids, trainlabels, X, Y, True)
    print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    # Train model
    print "training model..."
    stime = time.time()
    
    anova_filter = SelectKBest(f_classif, k=num_feats)
    anova_filter.fit(X, Y)
    X = anova_filter.transform(X)
    Y = to_categorical(Y)
    modelfile = args.prefix + "/" + args.name + ".model"
    if os.path.exists(modelfile):
        print "using pre-existing model at " + modelfile
        nn = load_model(modelfile)
    else:
        print "creating a new model"
        if model == "nn":
            nn = Sequential([
                    Dense(128, input_dim=num_feats),
                    Activation('relu'),
                    Dense(Y.shape[1]),
                    Activation('softmax'),
                ])
        
            nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
            nn.fit(X, Y)
        elif model == "lstm":
            nn = Sequential()
            #nn.add(Embedding(max_feat256, input_dim=200))
            nn.add(LSTM(256, input_dim=200, output_dim=Y.shape[1], activation='sigmoid', inner_activation='hard_sigmoid'))
            nn.add(Dropout(0.5))
            nn.add(Dense(1))
            nn.add(Activation('sigmoid'))
            nn.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
            nn.fit(X, Y, batch_size=16, nb_epoch=10)
            #score = model.evaluate(X_test, Y_test, batch_size=16)

        # Save the model
        print "saving the model..."
        nn.save(modelfile)
    
    etime = time.time()
    print "training took " + str(etime - stime) + " s"

    selected = anova_filter.get_support(True)
    print "features selected: "
    for i in selected:
        print "\t" + keys[i+2]

    # Test
    print "testing..."
    stime = time.time()
    testids = []
    testlabels = []
    testX = []
    testY = []
    testYenc = preprocess(args.testfile, testids, testlabels, testX, testY)
    testX = anova_filter.transform(testX)
    testY = to_categorical(testYenc)
    predictedY = nn.predict(testX)
    results = map_back(predictedY)
    predictedlabels = labelencoder.inverse_transform(results)
    etime = time.time()
    print "testing took " + str(etime - stime) + " s"
    print "testYenc: " + str(testYenc)
    print "results: " + str(results)

    # Calculate F1 score of results
    print "calculating scores..."
    precision = metrics.precision_score(testYenc, results)
    print "precision: " + str(precision)
    recall = metrics.recall_score(testYenc, results)
    print "recall: " + str(recall)
    f1score = metrics.f1_score(testYenc, results)
    print "f1: " + str(f1score)

    # Write results to a file
    output = open(args.outfile, 'w')
    for i in range(len(testids)):
        out = {}
        out['MG_ID'] = testids[i]
        out['Correct_ICD'] = testlabels[i]
        out['Predicted_ICD'] = predictedlabels[i]
        output.write(str(out) + "\n")
    output.close()

def preprocess(filename, ids, labels, x, y, trainlabels=False):
    global labelencoder

    # Read in the feature vectors
    starttime = time.time()
    print "preprocessing features..."
    f_encoders = {}
    types = []
    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)
            features = []
            for key in keys:
                if key == 'MG_ID':
                    ids.append(vector[key])
                    print "ID: " + vector[key]
                elif key == labelname:
                    labels.append(vector[key])
                elif key == "CL_type":
                    print "CL_type: " + vector[key]
                    types.append(vector[key])
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

    # Normalize features to 0 to 1
    preprocessing.minmax_scale(x, copy=False)
    endtime = time.time()
    print "preprocessing took " + str(endtime - starttime) + " s"
    return y

def map_back(results):
    #map = {0:1, 1:3, 2:4, 3:5, 4:6, 5:7, 6:8, 7:9, 8:10, 9:11, 10:13, 11:14, 12:15, 13:16, 14:17}
    output = []
    for x in range(len(results)):
        res = results[x]
        #print "res: " + str(res.tolist())
        val = numpy.argmax(res)
        output.append(val)
    return output


if __name__ == "__main__":main() 
