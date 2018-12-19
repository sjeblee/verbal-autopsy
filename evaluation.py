#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Get average scores from cross-validation
from __future__ import division

import argparse
#import statistics
import sys
from statsmodels.stats.contingency_tables import mcnemar

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in1', action="store", dest="results1")
    argparser.add_argument('--in2', action="store", dest="results2")
    args = argparser.parse_args()

    if not (args.in1 and args.in2):
        argparser.print_help(sys.stderr)
        #print("usage: ./score_crossval.py --in [.../crossval] --out [crossval_results.csv]")
        exit()

    run(args.results1, args.results2)

def run(in1, in2):
    pval = mcnemar_test(in1, in2)
    print('McNemar Test P-value:', pval)

def contingency_stats(infile):
    results = {}
    with open(infile, 'r') as f:
        for line in f.readlines():
            entry = eval(line)
            corr = entry['Correct_ICD']
            pred = entry['Predicted_ICD']
            if corr == pred:
                result = 1
            else:
                result = 0
            results[entry['MG_ID']] = result
    return results

def mcnemar_test(in1, in2):
    yes_yes = 0
    yes_no = 0
    no_yes = 0
    no_no = 0

    # Get individual predictions, mark as correct or not
    results1 = contingency_stats(in1)
    results2 = contingency_stats(in2)
    for key in results1.keys():
        res1 = results1[key]
        res2 = results2[key]
        if res1 + res2 == 2:
            yes_yes += 1
        elif res1 + res2 == 0:
            no_no += 1
        elif res1 == 1 and res2 == 0:
            yes_no += 1
        elif res1 == 0 and res2 ==1:
            no_yes +=1

    # Construct contingency table
    table = [[yes_yes, yes_no], [no_yes, no_no]]
    table = []
    stat, pval = mcnemar(table, exact=False, correction=True)
    return pval
