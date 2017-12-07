#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util
import xmltoseq

from lxml import etree
from sklearn_crfsuite import CRF, metrics
import argparse
import subprocess


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile):
        print "usage: ./model_crf.py --train [file.xml] --test [file.xml] (--out [file.xml])"
        exit()

    if args.outfile:
        run(args.trainfile, args.testfile, args.outfile)
    else:
        run(args.trainfile, args.testfile)

def run(trainfile, testfile, outfile=""):
    train_ids, train_seqs = get_seqs(trainfile)
    test_ids, test_seqs = get_seqs(testfile)
    trainx = [sent2features(s) for s in train_seqs]
    trainy = [sent2labels(s) for s in train_seqs]

    testx = [sent2features(s) for s in test_seqs]
    testy = [sent2labels(s) for s in test_seqs]

    # Train CRF
    print "training CRF..."
    crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
    crf.fit(trainx, trainy)

    # Ignore O tags for evaluation
    labels = list(crf.classes_)
    labels.remove('O')
    labels

    # Test CRF
    y_pred = crf.predict(testx)

    # Print metrics
    f1_score = metrics.flat_f1_score(testy, y_pred, average='weighted', labels=labels)
    print "F1: " + str(f1_score)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(
            testy, y_pred, labels=sorted_labels, digits=3
        ))

    # Convert crf output to xml tags
    test_dict = {}
    for x in range(len(test_ids)):
        rec_id = test_ids[x]
        rec_seq = zip((item[0] for item in test_seqs[x]), y_pred[x])
        test_dict[rec_id] = rec_seq
    xml_tree = xmltoseq.seq_to_xml(test_dict)

    # write the xml to file
    if len(outfile) > 0:
        print "Writing test output to xml file..."
        xml_tree.write(outfile)
        subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', outfile])
        subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', outfile])
            

def get_seqs(filename):
    ids = []
    narrs = []
    seqs = []
    
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    
    for child in root:
        narr = ""

        # Get the narrative text
        node = child.find("narr_timeml_simple")
        if node != None:
            rec_id = child.find("MG_ID").text
            narr = data_util.stringify_children(node).encode('utf-8')
            ids.append(rec_id)
            narrs.append(narr)

    for narr in narrs:
        narr_seq = xmltoseq.xml_to_seq(narr)
        seqs.append(narr_seq)

    return ids, seqs

def word2features(sent, i):
    word = sent[i][0]
    #postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'postag': postag,
        #'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            #'-1:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            #'+1:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def ispunc(input_string, start, end):
    punc = ' :;,./?'
    s = input_string[start:end]
    for char in s:
        if char not in punc:
            return False
    return True

if __name__ == "__main__":main()
