#!/usr/bin/python
# -*- coding: utf-8 -*-
# Calculate CSMF accuracy over multiple runs with dirichlet sampling and random assignment

from __future__ import division
from numpy.random import dirichlet
import argparse
import numpy
import random_assign

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-t', '--type', action="store", dest="type")
    argparser.add_argument('-n', '--num', action="store", dest="num")
    args = argparser.parse_args()

    if not args.type and args.num:
        print "usage: ./dirichlet_simulation.py --in [infile.txt] --type [adult/child/neonate] --num [number of simulations]"
        exit(1)

    run(args.type, args.num)

def run(arg_type, arg_num, arg_infile):
    arg_num = int(arg_num)
    print str(arg_num) + " iterations: " + arg_type

    all_labels = model.get_labels(arg_infile, 'cat_who')
    num = len(all_labels)/10 # size of the test set
    cat_map = list(set(all_labels))
    num_classes = len(labels)
    print "num classes: " + str(num_classes)
    print "classes: " + str(cat_map)
    prior = [1] * num_classes
    
    #adult_map = ['1','3','4','5','6','7','8','9','10','11','13','14','15','16','17']
    #child_map = ['1','3','4','5','6','7','8','9','10','11','12','13','14','15','16']
    #neonate_map = ['1','3','5','6','8','11','13','14','15']
    #adult_num = 9215
    #child_num = 1721
    #neonate_num = 465

    #cat_map = adult_map
    #adult_prior = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #child_prior = adult_prior
    #neonate_prior = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    #prior = adult_prior
    #num = adult_num
    #if arg_type == 'child':
    #    prior = child_prior
    #    cat_map = child_map
    #    num = child_num
    #elif arg_type == 'neonate':
    #    prior = neonate_prior
    #    cat_map = neonate_map
    #    num = neonate_num

    #records = get_recs(arg_infile)

    # Generate distributions
    s = numpy.random.dirichlet(prior, arg_num)#.transpose()
    csmfs = []
    for dist in s:
        gen_labels = []
        for x in range(len(prior)):
            # Generate distribution
            prop = dist[x]
            gen_num = int(round(prop * num))
            cat = cat_map[x]
            for y in range(gen_num):
                gen_labels.append(cat)
        correct_labels = random_assign.map_forward(gen_labels, cat_map)
        rand_labels = numpy.random.randint(0, len(prior), num)
        pred_labels = random_assign.map_back(rand_labels, cat_map)
            #print "gen_labels: " + str(gen_labels)
            #print "pred_labels: " + str(pred_labels)
            # Calculate CSMF accuracy of pred_labels and gen_labels
        acc = calculate_csmfa(correct_labels, rand_labels, len(prior))
        csmfs.append(acc)
    # Calculate mean CSMF accuracy
    mean_csmfa = sum(csmfs)/len(csmfs)
    print "mean CSMFA: " + str(mean_csmfa)

def get_recs(filename):
    recs = []
    with open(filename, 'r') as f:
        for line in f:
            recs.append(line.strip())
    #del recs[len(recs)-1]
    #del recs[0]
    return recs

def calculate_csmfa(correct, pred, num_classes):
    # Count the number of recs in each category
    n = len(correct)
    #print "calculating CSMF for " + str(n) + " records and " + str(num_classes) + " classes"
    corr_counts = []
    pred_counts = []
    for x in range(num_classes):
        corr_counts.append(0)
        pred_counts.append(0)

    for val in correct:
        corr_counts[val] = corr_counts[val] + 1

    for val2 in pred:
        pred_counts[val2] = pred_counts[val] + 1

    # Calculate CSMF accuracy
    csmf_pred = {}
    csmf_corr = {}
    csmf_corr_min = 1
    csmf_sum = 0
    for x in range(num_classes):
        num_corr = corr_counts[x]
        num_pred = pred_counts[x]
        csmf_c = num_corr/n
        csmf_p = num_pred/n
        csmf_corr[x] = csmf_c
        csmf_pred[x] = csmf_p
        #print "csmf for " + str(x) + " corr: " + str(csmf_c) + ", pred: " + str(csmf_p)
        if csmf_c < csmf_corr_min:
            csmf_corr_min = csmf_c
        csmf_sum = csmf_sum + abs(csmf_c - csmf_p)
    csmf_accuracy = 1 - (csmf_sum / (2 * (1 - csmf_corr_min)))
    return csmf_accuracy

if __name__ == "__main__":main()
