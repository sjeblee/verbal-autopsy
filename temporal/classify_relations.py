#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Find relations between events and times (and events/events?)

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
#import data_util

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
    run(args.model, args.pairtype, args.trainfile, "", testfile, outfile, relset)

def run(model, pairtype, trainfile, vecfile, testfile, outfile, relation_set='exact'):
    st = time.time()
    train_ids, train_feats, train_labels = extract_features(trainfile, relation_set, pairtype, True)

    print("train_feats: ", str(len(train_feats)), " train_labels: ", str(len(train_labels)))

    print("anova")
    ast = time.time()
    f_values = f_classif(train_feats, train_labels)
    print(str(f_values))
    print("anova took ", str(time.time()-ast), "s")

    mst = time.time()
    if model == 'svm':
        print("SVM model")
        classifier = SVC()
        classifier.fit(train_feats, train_labels)
    print("model training took ", str(time.time()-mst), "s")

    if testfile is not None:
        test_ids, test_feats, test_labels = extract_features(testfile, relation_set, pairtype)
        print("test_feats: ", str(len(test_feats)), " test_labels: ", str(len(test_labels)))

        # Test the model
        y_pred = classifier.predict(test_feats)
        print("y_pred: " + str(len(y_pred)))
        for x in range(10):
            print(str(x), " : ", str(y_pred[x]))

        # Score the output
        print(classification_report(test_labels, y_pred, target_names=relationencoder.classes_))
        #p = precision_score(test_labels, y_pred)
        #r = recall_score(test_labels, y_pred)
        #f1 = f1_score(test_labels, y_pred)
        #print("P: ", str(p), " R: ", str(r), " F1: ", str(f1))

        # TODO: use the TempEval3 evaluation script

        # TODO: write output to file
    print("total time: ", print_time(time.time()-st))

def extract_features(filename, relation_set='exact', pairtype='ee', train=False):
    print("extracting features: ", relation_set, pairtype)
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    features = []
    labels = []
    ids = []
    for child in root:
        id_node = child.find("record_id")
        rec_id = id_node.text
        node = child.find("narr_timeml_simple")
        if node != None:
            pairs, pair_labels = extract_pairs(node, relation_set, pairtype)
            for x in range(len(pair_labels)):
                labels.append(pair_labels[x])
                ids.append(rec_id)
                feats = pair_features(pairs[x], pairtype)
                features.append(feats)
    for x in range(10):
        print("features[", str(x), "]: ", str(features[x]))
        print("labels[", str(x), "]: ", str(labels[x]))

    # Normalize features
    num_feats = len(features[0])
    for y in range(num_feats):
        column = []
        for feat in features:
            column.append(feat[y])
        if train:
            global labelenc_map
            labelencoder = LabelEncoder()
            labelencoder.fit(column)
            labelenc_map[y] = labelencoder
        else:
            labelencoder = labelenc_map[y]
        norm_column = labelencoder.transform(column)
        for x in range(len(norm_column)):
            features[x][y] = norm_column[x]
    # Scale the features
    features_scaled = scale(features)
    print("features_scaled[0]: ", str(features_scaled[0]))

    # Encode the actual labels
    if train:
        global relationencoder
        relationencoder = LabelEncoder()
        relationencoder.fit(labels)
        print("labels: ", str(relationencoder.classes_))
    encoded_labels = relationencoder.transform(labels)

    print("feature extraction took ", print_time(time.time()-starttime))
    return ids, features_scaled, encoded_labels

''' Extract time-event pairs from xml data
'''
def extract_pairs(xml_node, relation_set='exact', pairtype='ee'):
    pairs = []
    labels = []
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    #print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))
    for event in events:
        if 'eid' not in event.attrib:
            print("no eid: ", etree.tostring(event))
        event_id = event.attrib['eid']

        # Get the position of the event
        event_position = xml_node.index(event)
        event.attrib['position'] = str(event_position)

        # Make a pair out of this event and all time phrases
        if pairtype == 'et' or pairtype == 'te':
            for time in times:
                time_id = time.attrib['tid']
                # Get the position of the time
                if not 'position' in time.attrib.keys():
                    time_position = xml_node.index(time)
                    time.attrib['position'] = str(time_position)
                pairs.append((event, time))
                labels.append(map_rel_type(none_label, relation_set))
        # Make pairs of events
        elif pairtype == 'ee':
            for event2 in events:
                pairs.append((event, event2))
                labels.append(map_rel_type(none_label, relation_set))

    # Find actual relations
    for tlink in tlinks:
        if pairtype == 'ee':
            if 'relatedToEventID' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                event2_id = tlink.attrib['relatedToEventID']
                rel_type = tlink.attrib['relType']
                mapped_type = map_rel_type(rel_type, relation_set)
                for x in range(len(labels)):
                    pair = pairs[x]
                    if pair[0].attrib['eid'] == event_id and pair[1].attrib['eid'] == event2_id:
                        labels[x] = mapped_type
        else:
            if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                time_id = tlink.attrib['relatedToTime']
                rel_type = tlink.attrib['relType']
                mapped_type = map_rel_type(rel_type, relation_set)
                for x in range(len(labels)):
                    pair = pairs[x]
                    if pair[0].attrib['eid'] == event_id and pair[1].attrib['tid'] == time_id:
                        labels[x] = mapped_type
    # TODO: undersample the NONE examples

    return pairs, labels

def map_rel_type(rel_type, relation_set):
    if relation_set == 'exact':
        return rel_type
    if relation_set == 'binary':
        if rel_type == none_label:
            return 0
        else:
            return 1
    elif relation_set == 'simple':
        if rel_type == 'NONE':
              return rel_type
        elif rel_type in ['BEFORE', 'IBEFORE']:
            return 'BEFORE'
        elif rel_type in ['AFTER', 'IAFTER']:
            return 'AFTER'
        else:
            return 'OVERLAP'

def pair_features(pair, pairtype):
    feats = []
    event = pair[0]
    
    event_feats = event_features(event)
    event_position = event.attrib['position']

    if pairtype == 'ee':
        event2 = pair[1]
        second_feats = event_features(event2)
        second_position = event2.attrib['position']
    else:
        time = pair[1]
        time_type = time.attrib['type']
        time_func = unk
        if 'temporalFunction' in time.attrib:
            time_func = time.attrib['temporalFunction']
        time_doc = unk
        if 'functionInDocument' in time.attrib:
            time_doc = time.attrib['functionInDocument']
        second_position = time.attrib['position']
        second_feats = [time_type, time_func, time_doc]

    distance = math.fabs(int(event_position) - int(second_position))
    feats = [distance] + event_feats + second_feats
    #print("feats: ", str(feats))
    return feats

def event_features(event):
    event_class = unk
    if 'class' in event.attrib:
        event_class = event.attrib['class']
    event_tense = unk
    if 'tense' in event.attrib:
        event_tense = event.attrib['tense']
    event_polarity = unk
    if 'polarity' in event.attrib:
        event_polarity = event.attrib['polarity']
    event_pos = unk
    if 'pos' in event.attrib:
        event_pos = event.attrib['pos']

    return [event_class, event_tense, event_polarity, event_pos]

def print_time(t):
    unit = "s"
    if t>60:
        t = t/60
        unit = "mins"
    if t>60:
        t = t/60
        unit = "hours"
    return str(t) + " " + unit

if __name__ == "__main__":main()
