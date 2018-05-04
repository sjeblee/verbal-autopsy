#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Extract temporal features

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util3

from gensim.models import KeyedVectors
from lxml import etree
from sklearn.feature_selection import f_classif
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.svm import SVC
import argparse
import math
import numpy
import time
import torch

# Local imports
import temporal_util as tutil

# Global variables
unk = "UNK"
none_label = "NONE"
labelenc_map = {}
relationencoder = None
undersample = 0.9

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-v', '--vectors', action="store", dest="vecfile")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    argparser.add_argument('-e', '--type', action="store", dest="pairtype")
    argparser.add_argument('-t', '--train', action="storeTrue", dest="train")
    args = argparser.parse_args()

    if not (args.infile and args.relset and args.pairtype):
        print("usage: ./extract_temporal_features.py --in [file_timeml.xml] --vectors [vecfile] --relset [binary/simple/exact] --type [ee/et]")
        exit()

    vecfile = None
    if args.vecfile:
        vecfile = args.vecfile
    relset = 'exact'
    if args.relset:
        relset = args.relset
    extract_features(args.infile, args.relset, args.pairtype, vecfile, args.train)

def extract_features(filename, relation_set='exact', pairtype='ee', vecfile=None, train=False):
    print("extracting features: ", relation_set, pairtype, "train: ", str(train))
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    features = []
    vec_features = []
    labels = []
    ids = []
    vec_model = None

    if vecfile is not None:
        print("Loading vectors: ", vecfile)
        vec_model = KeyedVectors.load_word2vec_format(vecfile, binary=True)
    
    for child in root:
        id_node = child.find("record_id")
        rec_id = id_node.text
        node = child.find("narr_timeml_simple")
        if node != None:
            pairs, pair_labels = extract_pairs(node, relation_set, pairtype)
            for x in range(len(pair_labels)):
                labels.append(pair_labels[x])
                ids.append(rec_id)
                feats, vec_feats = pair_features(pairs[x], pairtype, vec_model)
                features.append(feats)
                if vec_feats is not None:
                    vec_features.append(vec_feats)

    # Print the first few feature vectors as a sanity check
    for x in range(2):
        print("features[", str(x), "]: ", str(features[x]))
        print("vec_features[", str(x), "]: len ", str(len(vec_features[x])))
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

    # Merge the two feature sets
    features_final = []
    if len(vec_features) > 0:
        for z in range(len(vec_features)):
            #feats_sc = []
            #for item in features_scaled[z].tolist():
            #    feats_sc.append(numpy.asscalar(item))
            features_final.append(features_scaled[z].tolist() + vec_features[z])
    else:
        features_final = features_scaled

    # Encode the actual labels
    if train:
        global relationencoder
        relationencoder = LabelEncoder()
        relationencoder.fit(labels)
        print("labels: ", str(relationencoder.classes_))
    encoded_labels = relationencoder.transform(labels)

    print("feature extraction took ", tutil.print_time(time.time()-starttime))
    return ids, features_final, encoded_labels

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
                labels.append(tutil.map_rel_type(none_label, relation_set))
        # Make pairs of events
        elif pairtype == 'ee':
            for event2 in events:
                if event2.attrib['eid'] != event.attrib['eid']:
                    pairs.append((event, event2))
                    labels.append(tutil.map_rel_type(none_label, relation_set))

    # Find actual relations
    for tlink in tlinks:
        if pairtype == 'ee':
            if 'relatedToEventID' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                event2_id = tlink.attrib['relatedToEventID']
                rel_type = tlink.attrib['relType']
                mapped_type = tutil.map_rel_type(rel_type, relation_set)
                for x in range(len(labels)):
                    pair = pairs[x]
                    if pair[0].attrib['eid'] == event_id and pair[1].attrib['eid'] == event2_id:
                        labels[x] = mapped_type
        else:
            if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
                event_id = tlink.attrib['eventID']
                time_id = tlink.attrib['relatedToTime']
                rel_type = tlink.attrib['relType']
                mapped_type = tutil.map_rel_type(rel_type, relation_set)
                for x in range(len(labels)):
                    pair = pairs[x]
                    if pair[0].attrib['eid'] == event_id and pair[1].attrib['tid'] == time_id:
                        labels[x] = mapped_type
    # Undersample the NONE examples
    #print("undersampling NONE class: ", str(undersample))
    index = 0
    while index < len(labels):
        if labels[index] == 'NONE':
            #r = 0 # Temp for no NONE class
            r = numpy.random.random() # undersampling probability
            if r < undersample:
                del labels[index]
                del pairs[index]
                index = index-1
        index = index+1

    return pairs, labels

def pair_features(pair, pairtype, vec_model=None):
    feats = []
    vec_feats = None
    event = pair[0]

    # Structured features
    event_feats = event_features(event, vec_model)
    event_position = event.attrib['position']

    if pairtype == 'ee':
        event2 = pair[1]
        second_feats = event_features(event2, vec_model)
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

    # Word embedding features
    if vec_model is not None:
        first_emb = word_vector_features(event, vec_model)
        second_emb = word_vector_features(pair[1], vec_model)
        vec_feats = first_emb + second_emb

    distance = math.fabs(int(event_position) - int(second_position))
    feats = [distance] + event_feats + second_feats
    #print("feats: ", str(feats))
    return feats, vec_feats

def event_features(event, vec_model=None):
    second_feats = None
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
    feats = [event_class, event_tense, event_polarity, event_pos] 

    # Word embeddings
    #if vec_model is not None:
    #    second_feats = word_vector_features(event, vec_model)
    #    feats = feats + event_emb

    return feats

def get_relation_classes():
    if relationencoder is None:
        print("WARNING: relationencoder is None")
        return None
    else:
        return relationencoder.classes_

''' Get the average word embedding for the text of an element
    returns: a 1xd vector where d is the embedding dim
'''
def word_vector_features(element, vec_model):
    text = element.text
    words = text.strip().split(' ')

    if len(words) == 1:
        word = words[0]
        return get_vec(word, vec_model)
    else:
        vecs = []
        for word in words:
            vecs.append(get_vec(word, vec_model))
        avg_vec = torch.mean(torch.Tensor(vecs), 0).tolist()
        return avg_vec

def get_vec(word, model):
    dim = model.vector_size
    if word in model: #.wv.vocab:
        return model[word].tolist()
    else:
        return data_util3.zero_vec(dim)

if __name__ == "__main__":main()
