#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Find relations between events and times (and events/events?)

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
#import data_util
import model_library_torch
import extract_temporal_features

from keras.utils.np_utils import to_categorical
from lxml import etree
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC
import argparse
import math
import numpy
import time

# Global variables
unk = "UNK"
none_label = "NONE"
labelenc_map = {}
relationencoder = None

class Event:
    text = ""
    time = unk
    neg = False

    def __init__(self, text, time, neg=False):
        self.text = text
        self.time = time
        self.neg = neg

    def __str__(self):
        return self.time + ' : ' + self.to_text()

    def to_text(self):
        if self.neg:
            return 'no ' + self.text
        else:
            return self.text

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--train', action="store", dest="trainfile")
    argparser.add_argument('-t', '--test', action="store", dest="testfile")
    argparser.add_argument('-o', '--out', action="store", dest="outfile")
    argparser.add_argument('-v', '--vectors', action="store", dest="vecfile")
    argparser.add_argument('-m', '--model', action="store", dest="model")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    argparser.add_argument('-e', '--type', action="store", dest="pairtype")
    args = argparser.parse_args()

    if not (args.trainfile and args.model and args.pairtype):
        print("usage: ./timeline.py -m [svm] --train [file_timeml.xml] --vectors [vecfile] (--test [file.xml] --out [file.xml])")
        exit()

    testfile = None
    if args.testfile:
        testfile = args.testfile
    outfile = None
    if args.outfile:
        outfile = args.outfile
    relset = 'exact'
    if args.relset:
        relset = args.relset
    vecfile = None
    if args.vecfile:
        vecfile = args.vecfile
    run(args.model, args.pairtype, args.trainfile, vecfile, testfile, outfile, relset)

def run(model, pairtype, trainfile, vecfile, testfile, outfile, relation_set='exact'):
    st = time.time()
    train_ids, train_feats, train_labels = extract_temporal_features.extract_features(trainfile, relation_set, pairtype, train=True, vecfile=vecfile)

    print("train_feats: ", str(len(train_feats)), " train_labels: ", str(len(train_labels)))

    # ANOVA
    #print("anova")
    #ast = time.time()
    #f_values = f_classif(train_feats, train_labels)
    #print(str(f_values))
    #print("anova took ", str(time.time()-ast), "s")

    mst = time.time()
    if model == 'svm':
        print("SVM model")
        classifier = SVC()
        classifier.fit(train_feats, train_labels)
    elif model == 'nn':
        X = train_feats
        #Y = to_categorical(train_labels)
        Y = train_labels
        num_nodes = 100
        act = 'relu'
        epochs = 5
        classifier = model_library_torch.nn_model(X, Y, num_nodes, act, num_epochs=epochs)
    print("model training took ", str(time.time()-mst), "s")

    if testfile is not None:
        test_ids, test_feats, test_labels = extract_temporal_features.extract_features(testfile, relation_set, pairtype, vecfile=vecfile)
        print("test_feats: ", str(len(test_feats)), " test_labels: ", str(len(test_labels)))

        # Test the model
        if model == 'nn':
            y_pred = model_library_torch.test_nn(classifier, test_feats) # returns a list
        else:
            y_pred = classifier.predict(test_feats)
        print("y_pred: " + str(len(y_pred)))
        for x in range(10):
            print(str(x), " : ", str(y_pred[x]))

        # Score the output
        print(classification_report(test_labels, y_pred, target_names=extract_temporal_features.get_relation_classes()))
        #p = precision_score(test_labels, y_pred)
        #r = recall_score(test_labels, y_pred)
        #f1 = f1_score(test_labels, y_pred)
        #print("P: ", str(p), " R: ", str(r), " F1: ", str(f1))

        # TODO: use the TempEval3 evaluation script

        # TODO: write output to file
    print("total time: ", print_time(time.time()-st))

if __name__ == "__main__":main()
