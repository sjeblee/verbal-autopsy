#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Find relations between events and times (and events/events?)

import sys
sys.path.append('/u/sjeblee/research/git/verbal-autopsy')
#import model_library_torch
import extract_temporal_features
import temporal_util as tutil

sys.path.append('/u/sjeblee/research/git/learning2rank')
from rank import ListNet

from lxml import etree
#from sklearn.feature_selection import f_classif
#from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
#from sklearn.isotonic import IsotonicRegression
import argparse
import math
import numpy
import random
import time

# Global variables
debug = False
node_name = 'event_list_chrononet'
unk = "UNK"
none_label = "NONE"
labelenc_map = {}
relationencoder = None
pred_pairs_file = '/nbb/sjeblee/thyme/output/test_dctrel_predpairs_context.xml'

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--train', action="store", dest="trainfile")
    argparser.add_argument('-t', '--test', action="store", dest="testfile")
    argparser.add_argument('-o', '--out', action="store", dest="outfile")
    argparser.add_argument('-v', '--vectors', action="store", dest="vecfile")
    argparser.add_argument('-m', '--model', action="store", dest="model")
    argparser.set_defaults(model='listnet', vecfile='/u/sjeblee/research/vectors/wikipedia-pubmed-and-PMC-w2v.bin', outfile=None)
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile and args.outfile and args.vecfile):
        print("usage: ./chrononet.py -m [listnet] --train [file_timeml.xml] --vectors [vecfile] (--test [file.xml] --out [file.xml])")
        exit()

    run(args.model, args.trainfile, args.vecfile, args.testfile, args.outfile)

def run(model, trainfile, vecfile, testfile, outfile, relation_set='exact'):
    st = time.time()

    # TODO: run time/event tagging
    # For now we're using gold standard events and times

    # Extract temporal relation features
    train_ids, train_events, train_feats, train_ranks = extract_temporal_features.extract_rank_features(trainfile, train=True, vecfile=vecfile)
    print("train_feats: ", str(len(train_feats)), " train_ranks: ", str(len(train_ranks)))
    if debug: print("train_feats[0] len:", str(len(train_feats[0])), "train_ranks[0] len:", str(len(train_ranks[0])), str(type(train_feats[0])))

    if testfile is not None:
        test_ids, test_events, test_feats, test_ranks = extract_temporal_features.extract_rank_features(testfile, vecfile=vecfile)
        print("test_feats: ", str(len(test_feats)), " test_ranks: ", str(len(test_ranks)))
    print("feature extraction took", tutil.print_time(time.time()-st))

    mst = time.time()

    # ListNet Params
    modelfile = "listnet.model"
    epochs = 200

    # Train ListNet model
    listnet = ListNet.ListNet()
    # n_units1 = 128
    listnet.fit(train_feats, train_ranks, n_epoch=epochs, n_units1=256, n_units2=128, savemodelName=modelfile)
    print("model training took ", tutil.print_time(time.time()-mst))

    # For isotonic regression: concatenate all the training examples together
    '''
    iso_train = []
    iso_trainy = []
    for x in range(len(train_feats)):
        iso_train = iso_train + train_feats[x]
        iso_trainy = iso_trainy + train_ranks[x]
    isoX = numpy.asarray(iso_train)
    isoY = numpy.asarray(iso_trainy)
    print("isoX:", str(isoX.shape), "isoY:", str(isoY.shape))

    # Train isotonic regression model
    regression_model = IsotonicRegression(y_min=0, y_max=1)
    regression_model.fit(isoX, isoY)
    '''

    # Test the model
    if testfile is not None:

        pred_ranks = listnet.predict(test_feats)
        #if debug:
        print_sample("test_ranks", test_ranks)
        print_sample("listnet pred_ranks", pred_ranks)

        # Test isotonic regression model, one document at at time
        '''
        iso_pred_ranks = []
        for doc_feats in test_feats:
            isotestX = numpy.asarray(doc_feats)
            iso_pred_ranks.append(regression_model.predict(isotestX).tolist())
        '''

        # Extract pairs from ranked list
        print("Extracting pair relations...")
        rec_ids, true_pairs, true_relations = extract_temporal_features.extract_relation_pairs(testfile, train=True)

        # Load pairwise classifier results
        '''
        pc_ids, pc_pairs, pc_relations = extract_temporal_features.extract_relation_pairs(pred_pairs_file, train=True, nodename='pred_pair_relations')
        pairwise_recall = score_relation_pairs(pc_pairs, pc_relations, true_pairs, true_relations)
        print("METRICS: Pairwise relations")
        print("Pairwise accuracy:", str(pa))
        print("GPR:", str(pairwise_recall))
        #all_scores("Pairwise classification", rand_ranks, pc_pairs, pc_relations, test_ranks, true_pairs, true_relations, test_events)
        '''

        # Generate random ranks for comparison
        rand_ranks = random_ranks(test_events)
        rand_pairs, rand_relations = pair_relations(test_events, rand_ranks)
        all_scores("Random ranking", rand_ranks, rand_pairs, rand_relations, test_ranks, true_pairs, true_relations, test_events)

        # Span order metrics
        '''
        order_ranks = span_ranks(test_events)
        order_pairs, order_relations = pair_relations(test_events, order_ranks)
        all_scores("Span order", order_ranks, order_pairs, order_relations, test_ranks, true_pairs, true_relations, test_events)
        '''
        # Score the output as pairwise classification
        pred_pairs, pred_relations = pair_relations(test_events, pred_ranks)
        if debug: print_sample("pred_relations", pred_relations)
        all_scores("ListNet", pred_ranks, pred_pairs, pred_relations, test_ranks, true_pairs, true_relations, test_events)

        '''
        print("METRICS: isotonic regression")
        iso_mse, iso_pa = score_ranks(iso_pred_ranks, test_ranks)
        iso_pairs, iso_relations = pair_relations(test_events, iso_pred_ranks)
        iso_gold_recall = score_relation_pairs(iso_pairs, iso_relations, true_pairs, true_relations)
        print("MSE:", str(iso_mse))
        print("Pairwise accuracy:", str(iso_pa))
        print("GPR:", str(iso_gold_recall))
        '''
        
        # Oracle scores
        #all_scores("Oracle", test_ranks, true_pairs, true_relations, test_ranks, true_pairs, true_relations, test_events)

        # TODO: use the TempEval3 evaluation script

        # Write output to file
        write_output(testfile, outfile, test_ids, test_events, pred_ranks)

    print("total time: ", tutil.print_time(time.time()-st))

def all_scores(name, pred_ranks, pred_pairs, pred_relations, true_ranks, true_pairs, true_relations, test_events):
    print("METRICS:", name)
    #pred_pairs, pred_relations = pair_relations(test_events, pred_ranks)
    #if debug: print_sample("pred_relations", pred_relations)
    mse, pa = score_ranks(pred_ranks, true_ranks)
    print("MSE:", str(mse))
    print("POA (.001):", str(pa))
    pa2 = rank_pairwise_accuracy(pred_ranks, true_ranks, eps=0.01)
    print("POA (.01):", str(pa2))
    pa3 = rank_pairwise_accuracy(pred_ranks, true_ranks, eps=0.1)
    print("POA (.1):", str(pa3))
    pairwise_recall = score_relation_pairs(pred_pairs, pred_relations, true_pairs, true_relations)
    print("GPR:", str(pairwise_recall))

    #pred_pairs2, pred_relations2 = pair_relations(test_events, pred_ranks, eps=0.01)
    #pairwise_recall2 = score_relation_pairs(pred_pairs2, pred_relations2, true_pairs, true_relations)
    #print("GPR (.01):", str(pairwise_recall2))

def get_ordered_pairs(ranks):
    num = len(ranks)
    equal_pairs = []
    ordered_pairs = []
    for x in range(num):
        first = ranks[x]
        for y in range(num):
            if x != y:
                second = ranks[y]
                if first == second:
                    equal_pairs.append((x, y))
                elif first < second:
                    ordered_pairs.append((x,y))
    return equal_pairs, ordered_pairs


''' Generate all pair relations for ranked events
'''
def pair_relations(events, ranks, eps=0.0):
    pairs = []
    relations = []
    for n in range(len(events)):
        event_list = events[n]
        rank_list = ranks[n]
        doc_pairs = []
        doc_labels = []
        for x in range(len(event_list)):
            for y in range(len(event_list)):
                if x != y:
                    event1 = event_list[x]
                    event2 = event_list[y]
                    rank1 = float(rank_list[x])
                    rank2 = float(rank_list[y])
                    rel_type = 'OVERLAP'
                    if math.fabs(rank1-rank2) <= eps:
                        rel_type = 'OVERLAP'
                    elif rank1 < rank2:
                        rel_type = 'BEFORE'
                    elif rank1 > rank2:
                        rel_type = 'AFTER'
                    #print("rank pair", str(rank1), str(rank2), rel_type)
                    doc_pairs.append((event1, event2))
                    doc_labels.append(rel_type)
        pairs.append(doc_pairs)
        relations.append(doc_labels)
    return pairs, relations


''' Print a sample of a 2d list or array
'''
def print_sample(name, array, n=10):
    print(name, str(len(array)), str(len(array[0])), str(array[0][0:n]))


''' Generate random ranks for testing purposes
'''
def random_ranks(events, bin_ratio=0.25):
    ranks = []
    for n in range(len(events)):
        doc_ranks = []
        #num = math.ceil(len(events[n])*bin_ratio)
        for x in range(len(events[n])):
            doc_ranks.append(random.random())
        ranks.append(doc_ranks)
    return ranks


''' Calculate the mean squared error of the predicted ranks
'''
def rank_mse(pred_ranks, true_ranks):
    mse_scores = []
    for n in range(len(true_ranks)):
        num_samples = len(true_ranks[n])
        error_sum = 0
        for x in range(num_samples):
            error_sum += (true_ranks[n][x] - pred_ranks[n][x]) ** 2
        mse_scores.append(error_sum/float(num_samples))
    return numpy.average(numpy.asarray(mse_scores))


''' Calculate the pairwise accuracy of a listwise ranking
'''
def rank_pairwise_accuracy(pred_ranks, true_ranks, eps=0.001):
    accuracies = []
    for n in range(len(true_ranks)):
        pr = pred_ranks[n]
        se, so = get_ordered_pairs(true_ranks[n])
        num_pairs = len(so) + len(se)
        so_correct = 0
        se_correct = 0
        for pair in so:
            if pr[pair[0]] < pr[pair[1]]:
                so_correct += 1
        for pair in se:
            if math.fabs(pr[pair[0]] - pr[pair[1]]) <= eps:
                se_correct += 1
        accuracy = (so_correct + se_correct)/float(num_pairs)
        accuracies.append(accuracy)
    if len(accuracies) > 1:
        acc = numpy.average(numpy.asarray(accuracies))
    else:
        acc = accuracies[0]
    return acc


''' Score predicted ranks against correct ranks
'''
def score_ranks(pred_ranks, true_ranks, eps=0.001):
    if debug:
        print_sample('true_ranks', true_ranks)
        print_sample('pred_ranks:', pred_ranks)
    mse = rank_mse(pred_ranks, true_ranks)
    pa = rank_pairwise_accuracy(pred_ranks, true_ranks, eps)
    # TODO: relative list scores: Pairwise ordering accuracy
    return mse, pa


''' Score relations pairs against gold standard relation pairs
'''
def score_relation_pairs(pred_pairs, pred_labels, true_pairs, true_labels):
    doc_recalls = []
    doc_true_pairs = []
    doc_class_recalls = {}
    doc_class_recalls['BEFORE'] = []
    doc_class_recalls['AFTER'] = []
    doc_class_recalls['OVERLAP'] = []
    doc_class_totals = {}
    doc_class_totals['BEFORE'] = 0
    doc_class_totals['AFTER'] = 0
    doc_class_totals['OVERLAP'] = 0
    if debug: print('score_relation_pairs:', str(len(pred_pairs)), str(len(pred_labels)), str(len(true_pairs)), str(len(true_labels)))
    #assert(len(pred_labels) == len(true_labels))
    #assert(len(pred_pairs) == len(true_pairs))
    for x in range(len(pred_labels)):
        total = 0
        found = 0
        class_totals = {}
        class_totals['BEFORE'] = 0
        class_totals['AFTER'] = 0
        class_totals['OVERLAP'] = 0
        class_founds = {}
        class_founds['BEFORE'] = 0
        class_founds['AFTER'] = 0
        class_founds['OVERLAP'] = 0
        #print('tpairs:', str(len(true_pairs[x])))
        for y in range(len(true_pairs[x])):
            tpair = true_pairs[x][y]
            tlabel = tutil.map_rel_type(true_labels[x][y], 'simple')
            if debug: print('- tpair:', str_pair(tpair), 'tlabel:', str(tlabel))
            total += 1
            class_totals[tlabel] += 1
            #print('pred_pair[0]:', str(pred_pairs[x][0][0]), str(pred_pairs[x][0][1]))
            for z in range(len(pred_pairs[x])):
                ppair = pred_pairs[x][z]
                if tutil.are_pairs_equal(tpair, ppair):
                    plabel = pred_labels[x][z]
                    if debug: print("-- checking pair:", str_pair(ppair), str(plabel))
                    if tlabel == plabel:
                        found += 1
                        class_founds[tlabel] += 1
                        if debug: print('--- correct')
                    # Count before and before/overlap as the same since we're ranking on start time
                    elif tlabel == 'BEFORE/OVERLAP' and plabel == 'BEFORE':
                        if debug: print('--- correct (before/overlap)')
                        found += 1
                        class_founds[plabel] += 1
        if total == 0:
            print('WARNING: no reference relations found!')
            doc_recall = 0
        else:
            doc_recall = found/total
            for key in class_totals.keys():
                if class_totals[key] == 0:
                    val = 0.0
                else:
                    val = float(class_founds[key]) / class_totals[key]
                doc_class_recalls[key].append(val)
                doc_class_totals[key] += class_totals[key]
            doc_recalls.append(doc_recall)
            doc_true_pairs.append(total)

    # Calculate the weighted average recall
    avg_recall = numpy.average(doc_recalls, weights=doc_true_pairs)
    for key in doc_class_recalls.keys():
        avg_class_recall = numpy.average(numpy.asarray(doc_class_recalls[key]))
        print('Recall', key, str(avg_class_recall), 'num=', str(doc_class_totals[key]))
    
    return avg_recall

def str_pair(event_pair):
    return event_pair[0].attrib['eid'] + ' ' + event_pair[0].text + ' ' + event_pair[1].attrib['eid'] + ' ' + event_pair[1].text


''' Generate ranks based on order of mention in the text
    Note that this puts each event in its own bin
'''
def span_ranks(events):
    ranks = []
    for n in range(len(events)):
        doc_ranks = []
        num = len(events[n])
        for x in range(num):
            doc_ranks.append(float(x)/num)
        ranks.append(doc_ranks)
    return ranks


''' Write the output to a file
'''
def write_output(infile, outfile, ids, events, ranks):
    tree = etree.parse(infile)
    root = tree.getroot()
    id_name = "record_id"
    if debug: print('ids:', str(ids))
    for child in root:
        rec_id = child.find(id_name).text
        if rec_id in ids:
            index = ids.index(rec_id)
            list_node = etree.SubElement(child, node_name)
            for x in range(len(events[index])):
                event = events[index][x]
                rank = ranks[index][x]
                event.attrib['rank'] = str(rank)
                list_node.append(event)
    tree.write(outfile)

if __name__ == "__main__":main()
