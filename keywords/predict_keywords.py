#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')

from lxml import etree
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from keras.models import load_model
import argparse
import numpy
import os
import time

import data_util
import extract_features_temp as extract_features
import cluster_keywords
import model_library
#import model_seq

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--train', action="store", dest="trainfile")
    args = argparser.parse_args()

    if not (args.testfile and args.outfile and args.trainfile):
        print "usage: ./predict_keywords.py --train [file.xml] --test [file.xml] --out [outfile.xml]"
        exit()

    run(args.trainfile, args.testfile, args.outfile)

def run(trainfile, testfile, outfile):
    starttime = time.time()

    # Setup
    train_feat_file = trainfile + ".feats"
    train_kw_file = trainfile + ".kw_clusters"
    test_feat_file = testfile + ".feats"
    test_kw_file = testfile + ".kw_clusters"
    vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.50"

    # Extract word vector features and keyword vectors
    if not (os.path.exists(train_feat_file) and os.path.exists(test_feat_file)):
        extract_features.run(trainfile, train_feat_file, testfile, test_feat_file, arg_featurenames="narr_vec", arg_vecfile=vecfile)
    trainids, trainx = preprocess(train_feat_file, [], [], ["narr_vec"])
    testids, testx = preprocess(test_feat_file, [], [], ["narr_vec"])
    if not (os.path.exists(train_kw_file) and os.path.exists(test_kw_file)):
        extract_features.run(trainfile, train_kw_file, testfile, test_kw_file, arg_featurenames="kw_clusters", arg_vecfile=vecfile)
    trainyids, trainkws = preprocess(train_kw_file, [], [], ["keyword_clusters"])
    train_kw_dict = dict(zip(trainyids, trainkws))
    
    testyids, testkws = preprocess(test_kw_file, [], [], ["keyword_clusters"])
    test_kw_dict = dict(zip(testyids, testkws))

    # Match up the kw vectors with the trainids and testids
    trainy = []
    testy = []
    for x in range(len(trainids)):
        rec_id = trainids[x]
        trainy.append(train_kw_dict[rec_id])
    for y in range(len(testids)):
        rec_id = testids[y]
        testy.append(list(test_kw_dict[rec_id]))

    # keyword one-hot encoding
    #keywords = []
    #for item in trainy:
    #    for kw in item:
    #        keywords.append(kw)
    #labelencoder = model_seq.create_labelencoder(keywords)
    #trainy = model_seq.encode_labels(trainy, labelencoder)
    #testy = model_seq.encode_labels(testy, labelencoder)

    # keyword multi-hot encoding
    trainy = numpy.asarray(data_util.multi_hot_encoding(trainy, 299))
    testy = numpy.asarray(data_util.multi_hot_encoding(testy, 299))
    #print "trainy len: " + str(len(trainy))

    # Train and test the model
    print "trainy shape: " + str(trainy.shape)
    #output_seq_len = numpy.asarray(trainy).shape[1]
    #print "output_seq_len: " + str(output_seq_len)
    #print "trainx[0]: " + str(trainx[0])
    print "trainy[0]: " + str(trainy[0])

    # Seq2seq
    #model, encoder, decoder, output_dim = model_seq.train_seq2seq(trainx, trainy, True)
    #testy_pred = model_seq.predict_seqs(encoder, decoder, testx, output_seq_len, output_dim, True)

    # CNN
    modelfile = "cnn_keyword.model"
    if os.path.exists(modelfile):
        print "Using existing model"
        cnn = load_model(modelfile)
    else:
        print "Training new model..."
        cnn, x, y = model_library.rnn_model(trainx, trainy, 100, modelname='gru', num_epochs=15)
        cnn.save(modelfile)

    # Test
    testy_pred = cnn.predict(numpy.asarray(testx))
    #testy_pred_0 = data_util.map_to_multi_hot(testy_pred, 0.5)
    testy_pred = data_util.map_to_multi_hot(testy_pred)

    #for x in range(len(testy)):
    #    testy[x] = testy[x].tolist()
    testy = testy.tolist()
    print "testx[0]: " + str(testx[0])
    print "testy[0]: " + str(type(testy[0])) + " " + str(testy[0])
    print "tesy_pred[0]: " + str(type(testy_pred[0])) + " " + str(testy_pred[0])

    # Decode labels
    testy_pred_labels = data_util.decode_multi_hot(testy_pred)
    print "testy_pred_labels[0]: " + str(testy_pred_labels[0])

    clusterfile = "/u/sjeblee/research/va/data/datasets/mds+rct/train_adult_cat_spell.clusters"
    kw_pred_text = cluster_keywords.interpret_clusters(testy_pred_labels, clusterfile)
    kw_true_text = cluster_keywords.interpret_clusters(data_util.decode_multi_hot(testy), clusterfile)
    print "kw_pred_text[0]: " + str(kw_pred_text[0])
    print "kw_true_text[0]: " + str(kw_true_text[0])

    # Score results against nearest neighbor classifier
    print "Scores for 1 class (0.1 cutoff):"
    precision, recall, f1, micro_p, micro_r, micro_f1 = data_util.score_vec_labels(testy, testy_pred)
    print "Macro KW scores:"
    print "F1: " + str(f1)
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "Micro KW scores:"
    print "F1: " + str(micro_f1)
    print "precision: " + str(micro_p)
    print "recall: " + str(micro_r)
    
    # Save ouput to file
    #pred_dict = dict(zip(testids, testy_pred_labels))
    #output = open(outfile, 'w')
    #output.write(str(pred_dict))
    #output.close()
    cluster_keywords.write_clusters_to_xml(testfile, outfile, testids, testy_pred_labels)

    endtime = time.time()
    print "preprocessing took " + str(endtime - starttime) + " s"

def preprocess(filename, ids, x, feats, pad=False):
    # Read in the feature vectors
    max_seq_len = 0
    starttime = time.time()
    feats.append('MG_ID')
    print "preprocessing features: " + str(feats)
    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)
            features = []
            for key in feats:
                if key == 'MG_ID':
                    ids.append(vector[key])
                    #print "ID: " + vector[key]
                else:
                    print "key: " + key
                    # The feature matrix for word2vec can't have other features
                    features = vector[key]
                    #global max_seq_len
                    #max_seq_len = vector['max_seq_len']
                    #print "max_seq_len: " + str(max_seq_len)
                    if key == "keyword_clusters":
                        if features is None:
                            features = []
                        else:
                            features = features.split(',')
                        max_seq_len = max(max_seq_len, len(features))
                        #features = numpy.asarray(features, dtype='str')
                        print "feature len: " + str(len(features))
                    else:
                        features = numpy.asarray(features)
                        print "feature shape: " + str(features.shape)
                    x.append(features)

    # Pad keyword sequences
    print "max_seq_len: " + str(max_seq_len)
    if pad and (max_seq_len > 0):
        for num in range(len(x)):
            feats = x[num]
            while len(feats) < max_seq_len:
                feats.append("")
            features = numpy.asarray(feats, dtype='str')
            x[num] = features
    
    endtime = time.time()
    print "preprocessing took " + str(endtime - starttime) + " s"
    return ids, x                                                
                                                                                        
def ispunc(input_string, start, end):
    punc = ' :;,./?'
    s = input_string[start:end]
    for char in s:
        if char not in punc:
            return False
    return True

if __name__ == "__main__":main()
