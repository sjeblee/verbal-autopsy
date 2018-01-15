#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util
import xmltoseq

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from lxml import etree
from sklearn_crfsuite import CRF, metrics
import argparse
import subprocess


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile):
        print "usage: ./model_seq.py --train [file.xml] --test [file.xml] (--out [file.xml])"
        exit()

    if args.outfile:
        run(args.trainfile, args.testfile, args.outfile)
    else:
        run(args.trainfile, args.testfile)

def run(trainfile, testfile, outfile="", modelname="crf"):
    train_ids, train_seqs = get_seqs(trainfile)
    test_ids, test_seqs = get_seqs(testfile)

    if modelname == "crf":
        trainx = [sent2features(s) for s in train_seqs]
        trainy = [sent2labels(s) for s in train_seqs]
        testx = [sent2features(s) for s in test_seqs]
        testy = [sent2labels(s) for s in test_seqs]
    elif modelname == "seq2seq":
        trainx, trainy = get_feats(train_seqs)
        testx, testy = get_feats(test_seqs)
        print "TODO"

    print "train: " + str(len(trainx)) + " " + str(len(trainy))
    print "test: " + str(len(testx)) + " " + str(len(testy))

    # Split labels into time and event only
    #trainy_time, trainy_event = split_labels(trainy)
    #testy_time, testy_event = split_labels(testy)

    # Train model
    if modelname == "crf":
        model = train_crf(trainx, trainy)
        #crf_time = train_crf(trainx, trainy_time)
        #crf_event = train_crf(trainx, trainy_event)
    elif modelname == "seq2seq":
        model = train_seq2seq(trainx, trainy)
        #TODO: save model

    # Test CRF
    y_pred = model.predict(testx)
    #y_pred_time = crf_time.predict(testx)
    #y_pred_event = crf_event.predict(testx)

    # Print metrics
    print "testy: " + str(testy[0])
    print "y_pred: " + str(y_pred_event[0])
    #print "labels: " + str(labels[0])
    f1_score = score(testy, y_pred, labels)
    #f1_score_time = score(testy_time, y_pred_time, crf_time)
    #f1_score_event = score(testy_event, y_pred_event, crf_event)

    # Convert crf output to xml tags
    test_dict = {}
    for x in range(len(test_ids)):
        rec_id = test_ids[x]
        rec_seq = zip((item[0] for item in test_seqs[x]), y_pred[x])
        test_dict[rec_id] = rec_seq
    xml_tree = xmltoseq.seq_to_xml(test_dict, testfile)

    # write the xml to file
    if len(outfile) > 0:
        print "Writing test output to xml file..."
        xml_tree.write(outfile)
        subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', outfile])
        subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', outfile])

def train_crf(trainx, trainy):
    print "training CRF..."
    crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=100,
            all_possible_transitions=True
        )
    crf.fit(trainx, trainy)
    return crf

def train_seq2seq(trainx, trainy):
    print "NOT DONE YET"

    # TODO: fix input sequence params - need padding?
    num_encoder_tokens = 71
    num_decoder_tokens = 93
    latent_dim = 256
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = GRU(latent_dim, return_state=True)
    encoder_outputs, state_h = encoder(encoder_inputs)

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    decoder_gru = GRU(latent_dim, return_sequences=True)
    decoder_outputs = decoder_gru(decoder_inputs, initial_state=state_h)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data, epochs=10)

    return model

def score(testy, y_pred, crf):
    # Ignore O tags for evaluation
    labels = list(crf.classes_)
    labels.remove('O')
    f1_score = metrics.flat_f1_score(testy, y_pred, average='weighted', labels=labels)
    print "F1: " + str(f1_score)
    sorted_labels = sorted(labels, key=lambda name: (name[1:], name[0]))
    print(metrics.flat_classification_report(testy, y_pred, labels=sorted_labels, digits=3))
    return f1_score

def split_labels(y):
    t_labels = ['BT', 'IT']
    e_labels = ['BE', 'IE']
    y_time = []
    y_event = []
    for y_seq in y:
        time_seq = []
        event_seq = []
        for lab in y_seq:
            #print "lab: " + lab
            if lab in t_labels:
                time_seq.append(lab)
                event_seq.append('O')
            elif lab in e_labels:
                time_seq.append('O')
                event_seq.append(lab)
            else:
                time_seq.append(lab)
                event_seq.append(lab)
        y_time.append(time_seq)
        y_event.append(event_seq)
    return y_time, y_event

'''
   Get word vectors for each word and encode labels
   seqs: the sequence of pairs (word, label)
'''
def get_feats(seqs):
    vecfile = "" # TODO
    word2vec, dim = data_util.load_word2vec(vecfile)
    # TODO: create zero_vec
    feats = []
    labels = []
    for s in seqs:
        s_feats = []
        s_labels = []
        for pair in s:
            word = pair[0]
            vector = zero_vec
            if word in word2vec:
                vector = word2vec[word]
            s_feats.append(vector)
            s_labels.append(pair[1])
        feats.append(s_feats)
        labels.append(s_labels)
    # TODO: encode labels as numbers
    return feats, labels
                          
def get_seqs(filename):
    print "get_seqs " + filename
    ids = []
    narrs = []
    seqs = []
    
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    
    for child in root:
        narr = ""

        # Get the narrative text
        node = child.find("narr_timeml_simple")
        if node == None:
            narr_node = child.find("narrative")
            if narr_node == None:
                print "no narrative: " + data_util.stringify_children(child)
            else:
                rec_id = child.find("MG_ID").text
                #print "rec_id: " + rec_id
                ids.append(rec_id)
                narr = narr_node.text
                #print "narr: " + narr
                narrs.append(narr)
        else:
            rec_id = child.find("MG_ID").text
            #print "rec_id: " + rec_id
            narr = data_util.stringify_children(node).encode('utf-8')
            #print "narr: " + narr
            ids.append(rec_id)
            narrs.append(narr)

    for narr in narrs:
        narr_seq = xmltoseq.xml_to_seq(narr)
        seqs.append(narr_seq)

    return ids, seqs

def word2features(sent, i):
    word = sent[i][0]
    #postag = sent[i][1]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        #'postag': postag,
        #'postag[:2]': postag[:2],
    }
    if i > 0:
        word1 = sent[i-1][0]
        postag1 = sent[i-1][1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            #'-1:postag': postag1,
            #'-1:postag[:2]': postag1[:2],
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        word1 = sent[i+1][0]
        postag1 = sent[i+1][1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            #'+1:postag': postag1,
            #'+1:postag[:2]': postag1[:2],
        })
    else:
        features['EOS'] = True

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, label in sent]

def sent2tokens(sent):
    return [token for token, label in sent]

def ispunc(input_string, start, end):
    punc = ' :;,./?'
    s = input_string[start:end]
    for char in s:
        if char not in punc:
            return False
    return True

if __name__ == "__main__":main()
