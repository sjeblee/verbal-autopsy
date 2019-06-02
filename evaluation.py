#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Get average scores from cross-validation
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import sys
from lxml import etree
from sklearn.metrics.pairwise import cosine_similarity
from statsmodels.stats.contingency_tables import mcnemar

import word2vec
#from temporal import tag_symptoms

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

def kw_cosine_sim(infile, vecfile):

    # # TEMP
    sympfile="/u/sjeblee/research/data/va/resources/SYMP.csv" # Symptom file for symptom extraction
    chvfile="/u/sjeblee/research/data/va/resources/CHV_concepts_terms_flatfile_20110204.tsv"
    xmltree = etree.parse(infile)
    #xmltree = tag_symptoms.tag_symptoms(xmltree, sympfile, chvfile)
    #outfile = infile + '.symp'
    #xmltree.write(outfile)

    vectors, dim = word2vec.load(vecfile)
    print('dim:', str(dim))

    #xmltree = etree.parse(outfile)
    root = xmltree.getroot()
    rec_sims = []
    for child in root:
        kw_node = child.find('keywords_spell')
        symp_node = child.find('narr_symp')
        if kw_node is None or symp_node is None or kw_node.text is None or symp_node.text is None:
            print('WARNING: ones of the nodes is None!')
            rec_sims.append(0.0)
        else:
            kws = kw_node.text.split(',')
            symp = symp_node.text.split(',')

            kw_vecs = get_vectors(kws, vectors)
            symp_vecs = get_vectors(symp, vectors)
            #print('kw_vecs:', str(len(kw_vecs)), 'symp_vecs:', str(len(symp_vecs)))

            # Compute cosine similarity, pick the highest match for each symp
            sim_scores = []
            for symp_vec in symp_vecs:
                cosines = []
                for kw_vec in kw_vecs:
                    sim = cosine_similarity(symp_vec.reshape(1, -1), kw_vec.reshape(1, -1))
                    sim_score = sim[0][0]
                    cosines.append(sim_score)
                    print('sim score:', str(sim_score))
                best_cs = max(cosines)
                sim_scores.append(best_cs)
            rec_avg = numpy.average(numpy.asarray(sim_scores))
            print('rec avg cosine:', str(rec_avg))
            rec_sims.append(rec_avg)

    # Compute the average best cosine similarity
    avg_cosine = numpy.average(numpy.asarray(rec_sims))
    print('total avg cosine:', str(avg_cosine))
    return avg_cosine


def get_vectors(kw_list, vectors):
    kw_vecs = []
    for kw in kw_list:
        kw = kw.strip()
        #print('string:', kw)
        words = kw.split(' ')
        word_vecs = []
        for w in words:
            w = w.strip()
            if len(w) > 0:
                #print('word:', w)
                w_vec = numpy.array(word2vec.get(w, vectors))
                #print('w_vec:', str(w_vec.shape))
                word_vecs.append(w_vec)
        if len(word_vecs) > 0:
            word_array = numpy.array(word_vecs)
            kw_vecs.append(numpy.average(word_array, axis=0))
            #print('avg vec:', str(kw_vecs[-1]))
        else:
            print('WARNING: no words found!')
    return kw_vecs


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
