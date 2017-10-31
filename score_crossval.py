#!/usr/bin/python
# -*- coding: utf-8 -*-
# Get average scores from cross-validation
from __future__ import division

import argparse
import fnmatch
import os
import statistics

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--models', action="store",dest="models")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./score_crossval.py --in [.../crossval] --out [crossval_results.csv]"
        exit()

    models = "nb,rf,svm,nn"
    if args.models:
        models = args.models
    run(args.infile, args.outfile, models)

def run(arg_infile, arg_outfile, arg_models):
    crossval_dir = arg_infile
    models = arg_models.split(',')
    final_scores = {}
    global p, r, f1, pccc, csmfa
    p = 'p'
    r = 'r'
    f1 = 'f1'
    pccc = 'pccc'
    csmfa = 'csmfa'
    global metrics
    metrics = [p, r, f1, pccc, csmfa]

    for model in models:
        print "Getting scores for " + model
        neonate_scores = []
        child_scores = []
        adult_scores = []
        all_scores = []
        for x in range(0, 10):
            prefix = crossval_dir + "/" + model + "_" + str(x)
            for f in os.listdir(prefix):
                if fnmatch.fnmatch(f, 'test_neonate*.stats'):
                    n_scores = get_scores(prefix + "/" + f)
                    neonate_scores.append(n_scores)
                elif fnmatch.fnmatch(f, 'test_child*.stats'):
                    c_scores = get_scores(prefix + "/" + f)
                    child_scores.append(c_scores)
                elif fnmatch.fnmatch(f, 'test_adult*.stats'):
                    a_scores = get_scores(prefix + "/" + f)
                    adult_scores.append(a_scores)
                elif fnmatch.fnmatch(f, 'test_all*.stats'):
                    all_scores.append(get_scores(prefix + "/" + f))
        # Average the scores
        print "adult scores: " + str(len(adult_scores))
        print "child scores: " + str(len(child_scores))
        print "neonate scores: " + str(len(neonate_scores))
        print "all scores: " + str(len(all_scores))
        scores = {}
        scores['adult'] = []
        scores['child'] = []
        scores['neonate'] = []
        scores['all'] = []
        for metric in metrics:
            scores['adult'].append(avg_scores(adult_scores, metric))
            scores['child'].append(avg_scores(child_scores, metric))
            scores['neonate'].append(avg_scores(neonate_scores, metric))
            scores['all'].append(avg_scores(all_scores, metric))

        final_scores[model] = scores

    # write the stats to file
    output = open(arg_outfile, "w")
    output.write("model,precision,recall,f1,pccc,csmfa\n")

    # Adult
    output.write("adult\n")
    for model in models:
        output.write(model + ",")
        for x in range(0,5):
            output.write(str(final_scores[model]['adult'][x]) + ",")
        output.write("\n")

    # Child
    output.write("child\n")
    for model in models:
        output.write(model + ",")
        for x in range(0,5):
            output.write(str(final_scores[model]['child'][x]) + ",")
        output.write("\n")

    # Neonate
    output.write("neonate\n")
    for model in models:
        output.write(model + ",")
        for x in range(0,5):
            output.write(str(final_scores[model]['neonate'][x]) + ",")
        output.write("\n")

    # All
    output.write("all\n")
    for model in models:
        output.write(model + ",")
        for x in range(0,5):
            output.write(str(final_scores[model]['all'][x]) + ",")
        output.write("\n")

        
    output.close()

def get_scores(filename):
    scores = {}
    mnames = ['precision', 'recall', 'f1', 'pccc', 'csmf_accuracy']
    with open(filename, 'r') as f:
        for line in f:
            parts = line.split(',')
            if len(parts) > 1:
                name = parts[0]
                val = parts[1]
                if name == 'precision':
                    scores[p] = val
                elif name == 'recall':
                    scores[r] = val
                elif name == 'f1':
                    scores[f1] = val
                elif name == 'pccc':
                    scores[pccc] = val
                elif name == 'csmf_accuracy':
                    scores[csmfa] = val
    return scores

def avg_scores(scores, metric):
    avg_score = 0.0
    score = 0.0
    count = 0
    for entry in scores:
        score = score + float(entry[metric])
        count = count + 1
    if count > 0:
        avg_score = score / count
    return avg_score

def median_scores(scores, metric):
    metric_scores = []
    for entry in scores:
        metric_scores.append(float(entry[metric]))
    return statistics.median(metric_scores)

if __name__ == "__main__":main()
