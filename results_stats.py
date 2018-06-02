#!/usr/bin/python
# -*- coding: utf-8 -*-
# Generate distributional stats from results
from __future__ import division

import argparse
import numpy
from sklearn import metrics
import string

from scipy import stats

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--rank', action="store", dest="rank")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./results_stats.py --in [file.results] --out [outfile.csv]"
        exit()

    rank = 1
    if args.rank:
        rank = args.rank

    run(args.infile, args.outfile, rank)

def run(arg_infile, arg_outfile, arg_rank=1):
    labels_correct = {}
    labels_pred = {}
    correct = []
    predicted = []
    k = arg_rank     # Cat is considered correct if it's in the top k predicted categories

    # Get the xml from file
    with open(arg_infile, 'r') as f:
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
    num_classes = len(labels_correct.keys())
    print "n: " + str(n)
    print "num_classes: " + str(num_classes)

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

    # Edit by Yoona
    # Calculate KL Divergence (forward and reverse)
    dist_pred = []
    dist_corr = []
    keys = labels_correct.keys()

    print "Probability Distribution: "
    for key in keys:
        pred_prob = labels_pred[key] / n
        corr_prob = labels_correct[key] / n
        dist_pred.append(pred_prob)
        dist_corr.append(corr_prob)
        print "Correct probabilty for class " + str(key) + " is" + str(corr_prob)
        print "Predicted probabilty for class " + str(key) + " is" + str(pred_prob)

    kl_divergence_f = stats.entropy(dist_corr, dist_pred)
    kl_divergence_r = stats.entropy(dist_pred, dist_corr)

    # Calculate precision, recall, F1, and PCCC
    precision = metrics.precision_score(correct, predicted, average="weighted")
    recall = metrics.recall_score(correct, predicted, average="weighted")
    #total_recall = tp / (tp + fn)
    f1 = metrics.f1_score(correct, predicted, average="weighted")
    p_scores = metrics.precision_score(correct, predicted, average=None)
    r_scores = metrics.recall_score(correct, predicted, average=None)
    f1_scores = metrics.f1_score(correct, predicted, average=None)

    # PCCC
    pccc = ((tp/n) - (k/num_classes)) / (1 - (k/num_classes))

    # Confusion matrix
    confusion = metrics.confusion_matrix(correct, predicted, sorted(labels_correct.keys()))
    totals = []
    confusion_percent = []
    for row in confusion:
        total = 0
        for item in row:
            total = total + item
        row_percent = []
        totals.append(total)
        for item in row:
            item_percent = 0.0
            if total > 0:
                item_percent = float(item)/float(total)
            row_percent.append(item_percent)
        confusion_percent.append(row_percent)

    # Print metrics to terminal
    print "Per class metrics:"
    print "P, R, F1"
    for x in range(len(p_scores)):
        print str(p_scores[x]) + "," + str(r_scores[x]) + "," + str(f1_scores[x]) + "\n"

    print "Metrics:\n"
    print "p: " + str(precision) + "\n"
    print "r: " + str(recall) + "\n"
    #print "total_r: " + str(total_recall) + "\n"
    print "f1: " + str(f1) + "\n"
    print "pccc: " + str(pccc) + "\n"
    print "csmf accuracy: " + str(csmf_accuracy) + "\n"

    print "KL divergence forward: " + str(kl_divergence_f) + "\n"
    print "KL divergence reverse: " + str(kl_divergence_r) + "\n"
        
    # write the stats to file
    output = open(arg_outfile, "w")
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

    # Percentage confusion matrix
    output.write("confusion_matrix_percent")
    for key in keys:
        output.write("," + key)
    output.write("\n")
    for i in range(len(keys)):
        key = keys[i]
        row = confusion_percent[i]
        output.write(key)
        for j in range(len(row)):
            output.write("," + str(row[j]))
        output.write("," + str(totals[i]))
        #print "totals[i]: " + str(totals[i])
        output.write("\n")

    output.write("predicted distribution\nicd_cat,num_records\n")
    for key in labels_pred.keys():
        output.write(key + "," + str(labels_pred[key]) + "\n")
    output.write("correct distribution\nicd_cat,num_records\n")
    for key in labels_correct.keys():
        output.write(key + "," + str(labels_correct[key]) + "\n")
    output.close()

if __name__ == "__main__":main()
