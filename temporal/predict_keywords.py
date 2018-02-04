#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')

from lxml import etree
from sklearn.preprocessing import OneHotEncoder
import argparse
import numpy
import os
import time

#import data_util
import extract_features_temp as extract_features
import model_seq

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
    test_feat_file = testfile + ".feats"
    train_kw_file = trainfile + ".kw_words"
    test_kw_file = testfile + ".kw_words"
    vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.50"

    # Extract word vector features and keyword vectors
    if not (os.path.exists(train_feat_file) and os.path.exists(test_feat_file)):
        extract_features.run(trainfile, train_feat_file, testfile, test_feat_file, arg_featurenames="narr_vec", arg_vecfile=vecfile)
    trainids, trainx = preprocess(train_feat_file, [], [], ["narr_vec"])
    testids, testx = preprocess(test_feat_file, [], [], ["narr_vec"])
    if not (os.path.exists(train_kw_file) and os.path.exists(test_kw_file)):
        extract_features.run(trainfile, train_kw_file, testfile, test_kw_file, arg_featurenames="kw_words", arg_vecfile=vecfile)
    trainyids, trainkws = preprocess(train_kw_file, [], [], ["kw_words"])
    testyids, testkws = preprocess(test_kw_file, [], [], ["kw_words"])

    train_kw_dict = dict(zip(trainyids, trainkws))
    test_kw_dict = dict(zip(testyids, testkws))

    # Match up the kw vectors with the trainids and testids
    trainy = []
    testy = []
    for x in range(len(trainids)):
        rec_id = trainids[x]
        trainy.append(train_kw_dict[rec_id])
    for y in range(len(testids)):
        rec_id = testids[y]
        testy.append(test_kw_dict[rec_id])

    # keyword one-hot encoding
    keywords = []
    for item in trainy:
        for kw in item:
            keywords.append(kw)
    labelencoder = model_seq.create_labelencoder(keywords)
    trainy = model_seq.encode_labels(trainy, labelencoder)
    testy = model_seq.encode_labels(testy, labelencoder)

    # Train and test the model
    output_seq_len = numpy.array(testy).shape[1]
    print "output_seq_len: " + str(output_seq_len)
    model, encoder, decoder, output_dim = model_seq.train_seq2seq(trainx, trainy, True)
    testy_pred = model_seq.predict_seqs(encoder, decoder, testx, output_seq_len, output_dim, True)

    # Decode labels
    testy_temp = []
    for pred in testy_pred:
        lab = model_seq.decode_labels(pred, labelencoder)
        testy_temp.append(lab)
    testy_pred = testy_temp

    # TODO: figure out how to score this
    # TODO: figure out how to turn vectors back into words
    
    # Save ouput to file
    pred_dict = dict(zip(testids, testy_pred))
    output = open(outfile, 'w')
    output.write(str(pred_dict))
    output.close()

    endtime = time.time()
    print "preprocessing took " + str(endtime - starttime) + " s"

def preprocess(filename, ids, x, feats):
    # Read in the feature vectors
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
                    # The feature matrix for word2vec can't have other features
                    features = vector[key]
                    #global max_seq_len
                    #max_seq_len = vector['max_seq_len']
                    #print "max_seq_len: " + str(max_seq_len)
                    features = numpy.asarray(features)
                    print "feature shape: " + str(features.shape)
                    x.append(features)
    
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
