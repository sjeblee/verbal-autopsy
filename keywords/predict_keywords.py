#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy/temporal')

from collections import Counter
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
import extract_features
#import extract_features_temp as extract_features
import cluster_keywords
import model_library
import model_library_torch
import model_seq

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--num', action="store", dest="num_clusters")
    argparser.add_argument('--clusters', action="store", dest="clusterfile")
    argparser.add_argument('--model', action="store", dest="model")
    argparser.add_argument('--vectors', action="store", dest="vecfile")
    args = argparser.parse_args()

    if not (args.testfile and args.outfile and args.trainfile and args.num_clusters):
        print "usage: ./predict_keywords.py --train [file.xml] --test [file.xml] --out [outfile.xml] --num [num_clusters] --clusters [clusterfile] -- model [cnn/seq/encoder-decoder]"
        exit()

    run(args.trainfile, args.testfile, args.outfile, args.num_clusters, model=args.model, clusterfile=args.clusterfile, vecfile=args.vecfile)

def run(trainfile, testfile, outfile, num_clusters, model='seq', clusterfile=None, vecfile=None):
    starttime = time.time()

    # Setup
    train_feat_file = trainfile + ".feats"
    train_kw_file = trainfile + ".kw_clusters"
    test_feat_file = testfile + ".feats"
    test_kw_file = testfile + ".kw_clusters"
    if vecfile is None:
        vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
    #clusterfile = "/u/sjeblee/research/va/data/datasets/mds+rct/train_adult_cat_spell.clusters_km50"
    max_label = int(num_clusters) - 1

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

    # Calculate majority class
    kw_list = []
    for kw_items in trainkws:
        for item in kw_items:
            if item != "":
                kw_list.append(int(item))
    counter = Counter(kw_list)
    majority_kw = counter.most_common(1)[0][0]
    print "majority kw: " + str(majority_kw)
    majority_vec = data_util.multi_hot_encoding([[majority_kw]], max_label)[0]
    print "majority_vec: " + str(majority_vec)

    # Match up the kw vectors with the trainids and testids
    trainy = []
    testy = []
    for x in range(len(trainids)):
        rec_id = trainids[x]
        trainy.append(train_kw_dict[rec_id])
    for y in range(len(testids)):
        rec_id = testids[y]
        testy.append(list(test_kw_dict[rec_id]))

    print "trainy[0]: " + str(trainy[0])
    trainy_vecs = None
    trainy_phrases = None
    testy_vecs = None
    testy_phrases = None
    y_pred = []

    # keyword one-hot encoding
    if model == 'seq':
        keywords = []
        for item in trainy:
            for kw in item:
                keywords.append(kw)
        labelencoder = model_seq.create_labelencoder(keywords)
        trainy_vecs = model_seq.encode_labels(trainy, labelencoder)
        testy_vecs = model_seq.encode_labels(testy, labelencoder)
        print "trainy_vecs[0]: " + str(trainy_vecs[0])
    
    # keyword multi-hot encoding
    if model == 'cnn':
        trainy = data_util.multi_hot_encoding(trainy, max_label)
        testy = data_util.multi_hot_encoding(testy, max_label)

    if model == 'encoder-decoder':
        trainy_vecs, trainy_phrases = cluster_keywords.cluster_embeddings(trainy, clusterfile, vecfile, return_phrases=True)
        testy_vecs, testy_phrases = cluster_keywords.cluster_embeddings(testy, clusterfile, vecfile, return_phrases=True)

    # Train and test the model
    print "trainy size: " + str(len(trainy))
    #output_seq_len = numpy.asarray(trainy).shape[1]
    #print "output_seq_len: " + str(output_seq_len)
    #print "trainx[0]: " + str(trainx[0])
    print "trainy[0]: " + str(trainy[0])
    #print "trainy_phrases[0]: " + str(trainy_phrases[0])

    # Seq2seq - with sequences of one-hot encodings
    if model == 'seq':
        print "seq model"
        output_seq_len = 10
        nodes = 128
        model, encoder, decoder, output_dim = model_seq.train_seq2seq(trainx, trainy_vecs, nodes, True)
        y_pred = model_seq.predict_seqs(encoder, decoder, testx, output_seq_len, output_dim, True)
        testy_pred_labs = model_seq.decode_all_labels(y_pred, labelencoder)
        testy_pred_labels = [','.join(row) for row in testy_pred_labs] 
        #testy_pred_labels = cluster_keywords.embeddings_to_clusters(y_pred, clusterfile)
        #kw_true_text = testy_phrases
        testy = data_util.multi_hot_encoding(testy, max_label)
        testy_pred = data_util.multi_hot_encoding(testy_pred_labs, max_label)
        #testy_pred = data_util.map_to_multi_hot(y_pred)
        #testy_pred_labels = data_util.decode_multi_hot(testy_pred)
        print "testy_pred_labels[0]: " + str(testy_pred_labels[0])

    # Torch encoder-decoder
    elif model == 'encoder-decoder':
        print "torch encoder-decoder"
        output_seq_len = 10
        nodes = 100 # TODO: does this have to be the same as the word vector dim?
        encoder, decoder, output_dim = model_library_torch.encoder_decoder_model(trainx, trainy_vecs, nodes, num_epochs=1)
        y_pred = model_library_torch.test_encoder_decoder(encoder, decoder, testx, output_seq_len, output_dim)
        testy_pred_labels = cluster_keywords.embeddings_to_clusters(y_pred, clusterfile)
        kw_true_text = testy_phrases
        testy = data_util.multi_hot_encoding(testy, max_label)
        testy_pred = data_util.multi_hot_encoding(testy_pred_labels, max_label)

    # CNN
    elif model == 'cnn':
        #modelfile = "keyword_cnn_kwkm" + str(num_clusters) + ".model"
        modelfile = "/u/sjeblee/research/va/data/crossval_kw/gru_cnn_1/gru_cnn_1_adult.model"
        nodes = 100
        if os.path.exists(modelfile):
            print "Using existing model"
            cnn = load_model(modelfile)
        else:
            print "Training new model..."
            #cnn, x, y = model_library.rnn_model(trainx, trainy, 100, modelname='gru', num_epochs=15)
            #cnn, x, y = model_library.cnn_model(trainx, numpy.asarray(trainy), num_epochs=10, loss_func='mean_squared_error')
            cnn, x, y = model_library.stacked_model(trainx, [numpy.asarray(trainy)], nodes, models='gru_cnn', num_epochs=15, loss_func='mean_squared_error')
            cnn.save(modelfile)
        # Test
        #y_pred = cnn.predict(numpy.asarray(testx))
        
        # TEMP for multi model
        y_pred = cnn.predict(numpy.asarray(testx))[1].tolist()
        print "y_pred: " + str(len(y_pred))
        
        #testy_pred_0 = data_util.map_to_multi_hot(testy_pred, 0.5)
        testy_pred = data_util.map_to_multi_hot(y_pred)
        # Decode labels
        testy_pred_labels = data_util.decode_multi_hot(testy_pred)
        print "testy_pred_labels[0]: " + str(testy_pred_labels[0])
        kw_pred = [thing.split(',') for thing in testy_pred_labels]
        kw_true = [thing.split(',') for thing in data_util.decode_multi_hot(testy)]
        kw_emb, kw_pred_text = cluster_keywords.cluster_embeddings(kw_pred, clusterfile, vecfile, True)
        kw_true_emb, kw_true_text = cluster_keywords.cluster_embeddings(kw_true, clusterfile, vecfile, True)

        #testy = testy.tolist()
        #print "testx[0]: " + str(testx[0])
        print "testy[0]: " + str(len(testy[0])) + " " + str(testy[0])
        print "testy_pred[0]: " + str(len(testy_pred[0])) + " " + str(testy_pred[0])

    testy_pred_2 = data_util.map_to_multi_hot(y_pred, 0.2)
    testy_pred_3 = data_util.map_to_multi_hot(y_pred, 0.3)
    #kw_pred_text = cluster_keywords.interpret_clusters(testy_pred_labels, clusterfile)
    #kw_true_text = cluster_keywords.interpret_clusters(data_util.decode_multi_hot(testy), clusterfile)
    print "kw_pred_text[0]: " + str(kw_pred_text[0])
    print "kw_true_text[0]: " + str(kw_true_text[0])

    testy_pred_majority = []
    for x in range(len(testy_pred)):
        testy_pred_majority.append(majority_vec)

    #print "testy_pred: " + str(testy_pred)

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

    # Score results against nearest neighbor classifier
    print "Scores for 1 class (0.2 cutoff):"
    precision, recall, f1, micro_p, micro_r, micro_f1 = data_util.score_vec_labels(testy, testy_pred_2)
    print "Macro KW scores:"
    print "F1: " + str(f1)
    print "precision: " + str(precision)
    print "recall: " + str(recall)
    print "Micro KW scores:"
    print "F1: " + str(micro_f1)
    print "precision: " + str(micro_p)
    print "recall: " + str(micro_r)

    # Score results against nearest neighbor classifier
    print "Scores for 1 class (majority baseline):"
    precision, recall, f1, micro_p, micro_r, micro_f1 = data_util.score_vec_labels(testy, testy_pred_majority)
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
    cluster_keywords.write_clusters_to_xml(testfile, outfile, testids, kw_pred, kw_pred_text)

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
                    #print "key: " + key
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
                        #print "feature len: " + str(len(features))
                    else:
                        features = numpy.asarray(features)
                        #print "feature shape: " + str(features.shape)
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
