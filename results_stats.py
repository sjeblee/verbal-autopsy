#!/usr/bin/python
# -*- coding: utf-8 -*-
# Generate distributional stats from results
from __future__ import division

import argparse
import numpy
from sklearn import metrics
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--rank', action="store", dest="rank")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./results_stats.py --in [file.results] --out [outfile.csv]"
        exit()

    labels_correct = {}
    labels_pred = {}
    correct = []
    predicted = []
    k = 1     # Cat is considered correct if it's in the top k predicted categories

    if args.rank:
        k = int(args.rank)

    # Get the xml from file
    with open(args.infile, 'r') as f:
        for line in f:
            res = eval(line)
            pred = res['Predicted_ICD']
            predicted.append(pred)
            cor = res['Correct_ICD']
            correct.append(cor)

            if labels_correct.has_key(cor):
                labels_correct[cor] = labels_correct[cor] + 1
            else:
                labels_correct[cor] = 1

            if labels_pred.has_key(pred):
                labels_pred[pred] = labels_pred[pred] + 1
            else:
                labels_pred[pred] = 1

    tp = 0    # True positives
    fn = 0    # False negatives
    n = len(correct)
    print "n: " + str(n)

    # Calculate tp
    for x in range(n):
        cor = correct[x]
        pred = predicted[x]
        if pred == cor:
            tp = tp +1
        else:
            fn = fn +1

    # Calculate CSMF accuracy
    csmf_pred = {}
    csmf_corr = {}
    csmf_corr_min = 1
    csmf_sum = 0
    for key in labels_correct.keys():
        if not labels_pred.has_key(key):
            labels_pred[key] = 0
        num_corr = labels_correct[key]
        num_pred = labels_pred[key]
        csmf_c = num_corr/n
        csmf_p = num_pred/n
        csmf_corr[key] = csmf_c
        csmf_pred[key] = csmf_p
        #print "csmf for " + key + " corr: " + str(csmf_c) + ", pred: " + str(csmf_p)
        if csmf_c < csmf_corr_min:
            csmf_corr_min = csmf_c
        csmf_sum = csmf_sum + abs(csmf_c - csmf_p)

    csmf_accuracy = 1 - (csmf_sum / (2 * (1 - csmf_corr_min)))

    # Calculate precision, recall, F1, and PCCC
    precision = metrics.precision_score(correct, predicted, average="weighted")
    recall = metrics.recall_score(correct, predicted, average="weighted")
    total_recall = tp / (tp + fn)
    f1 = metrics.f1_score(correct, predicted, average="weighted")

    # PCCC
    pccc = ((tp/n) - (k/n)) / (1 - (k/n))

    # Confusion matrix
    confusion = metrics.confusion_matrix(correct, predicted, sorted(labels_correct.keys()))

    # Print metrics to terminal
    print "Metrics:\n"
    print "p: " + str(precision) + "\n"
    print "r: " + str(recall) + "\n"
    print "total_r: " + str(total_recall) + "\n"
    print "f1: " + str(f1) + "\n"
    print "pccc: " + str(pccc) + "\n"
    print "csmf accuracy: " + str(csmf_accuracy) + "\n"
        
    # write the stats to file
    output = open(args.outfile, "w")
    output.write("precision," + str(precision) + "\nrecall," + str(recall) + "\nf1," + str(f1) + "\npccc," + str(pccc) + "\ncsmf_accuracy," + str(csmf_accuracy) + "\n")

    output.write("confusion_matrix")
    keys = sorted(labels_correct.keys())
    for key in keys:
        output.write("," + key)
    output.write("\n")
    for i in range(len(keys)):
        key = keys[i]
        row = confusion[i]
        output.write(key)
        for j in range(len(row)):
            output.write("," + str(row[j]))
        output.write("\n")

    output.write("predicted distribution\nicd_cat,num_records\n")
    for key in labels_pred.keys():
        output.write(key + "," + str(labels_pred[key]) + "\n")
    output.write("correct distribution\nicd_cat,num_records\n")
    for key in labels_correct.keys():
        output.write(key + "," + str(labels_correct[key]) + "\n")
    output.close()

if __name__ == "__main__":main()
