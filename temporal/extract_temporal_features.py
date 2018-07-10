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
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from xml.sax.saxutils import unescape
import argparse
import math
import numpy
import time
import torch

# Local imports
import temporal_util as tutil

# Parameters
undersample = 1
debug = False
context_size = 3

# Global variables
listname = "event_list"
unk = "UNK"
none_label = "NONE"
labelenc_map = {}
relationencoder = None
scaler = None

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

def extract_rank_features(filename, relation_set='exact', vecfile=None, train=False):
    ids, events, features, labels = extract_features(filename, relation_set, vecfile=vecfile, train=train, rank_features=True)

    print("extract_features:", str(len(ids)), str(len(events)), str(len(features)), str(len(labels)))
    # TODO: include the document creation time?

    # Collapse ids and features into a 3-d list
    new_ids = []
    new_feats = []
    new_labels = []
    new_events = []
    prev_id = ids[0]
    new_feat_seq = []
    new_label_seq = []
    new_event_seq = []
    for x in range(len(ids)):
        #print("- x:", str(x), "id:", str(ids[x]))
        if ids[x] != prev_id:
            new_ids.append(prev_id)
            prev_id = ids[x]
            event_seq, feat_seq, label_seq = sort_seqs(new_event_seq, new_feat_seq, new_label_seq)
            new_events.append(event_seq)
            new_feats.append(feat_seq)
            new_labels.append(label_seq)
            new_feat_seq = []
            new_label_seq = []
            new_event_seq = []
        new_feat_seq.append(features[x])
        new_label_seq.append(labels[x])
        new_event_seq.append(events[x])
    # Save the last sequence
    event_seq, feat_seq, label_seq = sort_seqs(new_event_seq, new_feat_seq, new_label_seq)
    new_ids.append(prev_id)
    new_events.append(event_seq)
    new_feats.append(feat_seq)
    new_labels.append(label_seq)

    if debug: print("extract_rank_features:", str(len(new_ids)), str(len(new_events)), str(len(new_feats)), str(len(new_labels)))
    return new_ids, new_events, new_feats, new_labels

def sort_seqs(events, feats, labels):
    #if debug:
    #    print("new_event_seq:", str(len(events)))
    #    print("new_feat_seq:", str(len(feats)))
    #    print("new_label_seq:", str(len(labels)))

    # Scale the rank numbers
    labels = scale_ranks(labels)
    spans = []
    for event in events:
        spans.append(event.attrib['span'].split(',')[0])

    zipped = list(zip(spans, events, feats, labels))
    zipped = sorted(zipped, key=lambda x: x[0])
    spans, ev, fe, la = zip(*zipped)
    return list(ev), list(fe), list(la)

def extract_features(filename, relation_set='exact', pairtype='ee', vecfile=None, train=False, rank_features=False, limit=None):
    if debug: print("extracting features: ", relation_set, pairtype, "train: ", str(train))
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    features = []
    vec_features = []
    labels = []
    all_events = []
    ids = []
    all_pairs = []
    vec_model = None
    dropped = 0
    records = 0

    if rank_features:
        nodename = "event_list"
    else:
        nodename = "narr_timeml_simple"

    if vecfile is not None:
        print("Loading vectors: ", vecfile)
        vec_model = KeyedVectors.load_word2vec_format(vecfile, binary=True)
    
    for child in root:
        if limit is not None and (records >= limit):
            dropped += 1
            continue
        id_node = child.find("record_id")
        rec_id = id_node.text
        #if debug: print("rec_id:", rec_id)
        records += 1
        node = child.find(nodename)
        narrative = child.find('narrative').text
        #print("node is None:", str(node is None))
        if node is not None:
            try:
                if rank_features:
                    node = etree.fromstring(etree.tostring(node).decode('utf8'))
                else:
                    node = etree.fromstring('<' + nodename + '>' + data_util3.stringify_children(node).encode('utf8').decode('utf8') + '</' + nodename + '>')
            except etree.XMLSyntaxError as e:
                dropped += 1
                position = e.position[1]
                print("XMLSyntaxError at ", e.position, str(e), data_util3.stringify_children(node)[position-5:position+5])

            if rank_features:
                events, feats, vec_feats, ranks = extract_event_feats(node, vec_model, narrative)
                for x in range(len(feats)):
                    ids.append(rec_id)
                    features.append(feats[x])
                    labels.append(ranks[x])
                    vec_features.append(vec_feats[x])
                    all_events.append(events[x])
            else:
                #if train:
                #    us = .999
                #else:
                #    us = 0
                us = 1
                pairs, pair_labels = extract_pairs(node, relation_set, pairtype, train, under=us)
                for x in range(len(pair_labels)):
                    labels.append(pair_labels[x])
                    ids.append(rec_id)
                    feats, vec_feats = pair_features(pairs[x], pairtype, vec_model, narrative)
                    features.append(feats)
                    if pairtype == 'ee':
                        all_pairs.append((pairs[x][0].attrib['eid'], pairs[x][1].attrib['eid']))
                    if vec_feats is not None:
                        vec_features.append(vec_feats)
        else:
            #print("node is None!")
            dropped += 1

    # Print the first few feature vectors as a sanity check
    print("records:", str(records))
    print("examples:", str(len(features)))
    print("dropped:", str(dropped))
    for x in range(3):
        print("features[", str(x), "]: ", str(features[x]))
        #print("vec_features[", str(x), "]: len ", str(len(vec_features[x])))
        print("labels[", str(x), "]: ", str(labels[x]))

    # Normalize features
    num_feats = len(features[0])
    global labelenc_map
    for y in range(1,num_feats):
        if type(features[0][y]) is str: # Only encode string features
            column = []
            for feat in features:
                column.append(feat[y])
            if train:
                if debug: print('training labelencoder')
                global labelenc_map
                labelencoder = LabelEncoder()
                labelencoder.fit(column)
                labelenc_map[y] = labelencoder
            else:
                labelencoder = labelenc_map[y]
            norm_column = labelencoder.transform(column)
            for x in range(len(norm_column)):
                features[x][y] = norm_column[x]
    for x in range(1):
        print("encoded features[", str(x), "]: ", str(features[x]))

    # Scale the features
    global scaler
    if train:
        scaler = MinMaxScaler()
        scaler.fit(features)
    features_scaled = scaler.transform(features)
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

    # Encode the pairwise labels (but not rank labels)
    if not rank_features:
        if train:
            global relationencoder
            relationencoder = LabelEncoder()
            relationencoder.fit(labels)
            if debug: print("labels: ", str(relationencoder.classes_))
        encoded_labels = relationencoder.transform(labels)
        num_classes = len(relationencoder.classes_)

        # One-hot encoding
        #onehot_labels = []
        #for lab in encoded_labels:
        #    onehot_lab = numpy.zeros(num_classes).tolist()
        #    onehot_lab[lab] = 1
        #    onehot_labels.append(onehot_lab)
        labels = encoded_labels

    # Free the word embedding model
    del vec_model

    print("feature extraction took ", tutil.print_time(time.time()-starttime))
    if rank_features:
        return ids, all_events, features_final, labels
    else:
        return ids, all_pairs, features_final, labels

''' Extract event features
'''
def extract_event_feats(xml_node, vec_model, narrative):
    events = xml_node.xpath("EVENT")
    if debug: print('extract_event_feats: events:', str(len(events)))
    feats = []
    vec_feats = []
    ranks = []
    for event in events:
        event_feats = event_features(event, vec_model, narr=narrative)
        #print('- event_features:', str(len(event_features)))
        event_vec = word_vector_features(event, vec_model)
        rank = int(event.attrib['rank'])
        feats.append(event_feats)
        #print('- feats:', str(len(feats)))
        vec_feats.append(event_vec)
        ranks.append(rank)
    return events, feats, vec_feats, ranks


''' Extract relation pairs
'''
def extract_relation_pairs(infile, relation_set='exact', pairtype='ee', train=False, nodename='narr_timeml_simple'):
    if debug: print("extracting relation pairs: ", relation_set, pairtype, "train: ", str(train))
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()
    all_labels = []
    all_pairs = []
    ids = []
    records = 0
    #nodename = name
    global undersample
    undersample = 1

    for child in root:
        id_node = child.find("record_id")
        rec_id = id_node.text
        records += 1
        node = child.find(nodename)
        try:
            node = etree.fromstring('<' + nodename + '>' + data_util3.stringify_children(node).encode('utf8').decode('utf8') + '</' + nodename + '>')
        except etree.XMLSyntaxError as e:
            #dropped += 1
            position = e.position[1]
            print("XMLSyntaxError at ", e.position, str(e), data_util3.stringify_children(node)[position-5:position+5])
        if node is not None:
            ids.append(rec_id)
            pairs, pair_labels = extract_pairs(node, relation_set, pairtype, train)
            all_pairs.append(pairs)
            all_labels.append(pair_labels)

    return ids, all_pairs, all_labels


''' Extract time-event pairs from xml data
'''
def extract_pairs(xml_node, relation_set='exact', pairtype='ee', train=False, under=1):
    pairs = []
    labels = []
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    if debug: print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))
    for event in events:
        if 'eid' not in event.attrib:
            print("no eid: ", etree.tostring(event))
        event_id = event.attrib['eid']

        # Get the position of the event
        event_position = xml_node.index(event)
        event.attrib['position'] = str(event_position)

        # Make a pair out of this event and all time phrases
        if pairtype == 'et' or pairtype == 'te':
            for timex in times:
                time_id = timex.attrib['tid']
                # Get the position of the time
                if not 'position' in timex.attrib.keys():
                    time_position = xml_node.index(timex)
                    timex.attrib['position'] = str(time_position)
                pairs.append((event, time))
                labels.append(tutil.map_rel_type(none_label, relation_set))
        # Make pairs of events
        elif pairtype == 'ee':
            for event2 in events:
                if event2.attrib['eid'] != event.attrib['eid']:
                    pairs.append((event, event2))
                    labels.append(tutil.map_rel_type(none_label, relation_set))

    #print("pairs:", str(len(pairs)))
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
    # Undersample the NONE examples for training or all
    do_undersample = True
    if do_undersample:
        if debug: print("undersampling NONE class: ", str(under))
        index = 0
        while index < len(labels):
            if labels[index] == none_label:
                #r = 0 # Temp for no NONE class
                r = numpy.random.random() # undersampling probability
                if r < under:
                    del labels[index]
                    del pairs[index]
                    index = index-1
            index = index+1
        if debug: print("pairs after undersample:", str(len(pairs)))

    return pairs, labels


''' Get pair features
'''
def pair_features(pair, pairtype, vec_model=None, narrative=None):
    #print("pair_features:", pairtype)
    time_fnames = ['type', 'temporalFunction', 'functionInDocument']
    feats = []
    vec_feats = None
    event = pair[0]

    # Structured features
    event_feats = event_features(event, vec_model, narrative)
    event_position = float(event.attrib['position'])/len(narrative)

    if pairtype == 'ee':
        event2 = pair[1]
        second_feats = event_features(event2, vec_model, narrative)
        second_position = float(event2.attrib['position'])/len(narrative)
    else:
        time = pair[1]
        second_feats = []
        for fname in time_fnames:
            if fname in time.attrib:
                second_feats.append(time.attrib[fname])
            else:
                second_feats.append(unk)
        second_position = time.attrib['position']

    # Word embedding features
    if vec_model is not None:
        first_emb = word_vector_features(event, vec_model)
        second_emb = word_vector_features(pair[1], vec_model)
        vec_feats = first_emb + second_emb

    distance = int(event_position) - int(second_position)
    feats = [distance] + event_feats + second_feats
    #print("feats: ", str(feats))
    return feats, vec_feats


def event_features(event, vec_model=None, narr=None):
    second_feats = None
    event_class = unk
    feats = []
    featurenames = ['class','tense','pos','polarity','contextualmodality','contextualaspect','type']
    # Add start span
    span = event.attrib['span'].split(',')
    span_start = int(span[0])
    span_end = int(span[1])

    # Add start span as a features
    feats.append(span_start)

    for fname in featurenames:
        if fname in event.attrib:
            feats.append(event.attrib[fname])
        else:
            feats.append(unk)

    # Add context features
    if narr is not None and vec_model is not None:
        prev_text = narr[0:span_start].strip().lower()
        next_text = narr[span_end:].strip().lower()
        prev_words = prev_text.split()
        next_words = next_text.split()
        prev_context = []
        next_context = []
        dim = vec_model.vector_size
        zero_vec = numpy.zeros(dim).tolist()

        for w in range(context_size):
            prev_index = len(prev_words)-1 - w
            next_index = w
            if prev_index < 0:
                prev_context = zero_vec + prev_context
            else:
                prev_context = get_vec(prev_words[prev_index], vec_model)
            if next_index >= len(next_words):
                next_context = next_context + zero_vec
            else:
                next_context = next_context + get_vec(next_words[next_index], vec_model)

        context_vec = prev_context + next_context
        feats = feats + context_vec
    return feats

def get_relation_classes():
    if relationencoder is None:
        print("WARNING: relationencoder is None")
        return None
    else:
        return relationencoder.classes_

''' Get the encoder used for relation class labels
'''
def get_relation_encoder():
    return relationencoder

''' Scale event ranks
'''
def scale_ranks(ranks):
    max_rank = max(ranks)
    sr = []
    for rank in ranks:
        sr.append(float(rank)/float(max_rank))
    return sr

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
