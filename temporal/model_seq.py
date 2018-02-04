#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util
import xmltoseq

from keras.models import Model
from keras.layers import Input, GRU, LSTM, Dense, Dropout
from keras.preprocessing.sequence import pad_sequences
from lxml import etree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn_crfsuite import CRF, metrics
import argparse
import numpy
import subprocess

global labelencoder, onehotencoder, label_set, max_seq_len, num_labels

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--model', action="store", dest="model")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile):
        print "usage: ./model_seq.py --train [file.xml] --test [file.xml] (--out [file.xml])"
        exit()
        

    if args.outfile and args.model:
        run(args.trainfile, args.testfile, args.outfile, args.model)
    elif args.outfile:
        run(args.trainfile, args.testfile, args.outfile)
    else:
        run(args.trainfile, args.testfile)

def run(trainfile, testfile, outfile="", modelname="crf"):
    train_ids, train_seqs = get_seqs(trainfile, split_sents=True)
    test_ids, test_seqs = get_seqs(testfile, split_sents=True)

    if modelname == "crf":
        trainx = [sent2features(s) for s in train_seqs]
        trainy = [sent2labels(s) for s in train_seqs]
        testx = [sent2features(s) for s in test_seqs]
        testy = [sent2labels(s) for s in test_seqs]
    elif modelname == "nn":
        extra_seqs = get_seqs('/u/sjeblee/research/data/TimeBank/timebank_all_timeml_simple.xml', split_sents=True)
        train_seqs = extra_seqs + train_seqs
        trainx, trainy = get_feats(train_seqs, True)
        testx, testy = get_feats(test_seqs)

    print "train: " + str(len(trainx)) + " " + str(len(trainy))
    print "test: " + str(len(testx)) + " " + str(len(testy))

    # Split labels into time and event only
    trainy_time, trainy_event = split_labels(trainy)
    testy_time, testy_event = split_labels(testy)

    # Train model
    model = None
    encoder_model = None
    decoder_model = None
    y_pred = []
    if modelname == "crf":
        model = train_crf(trainx, trainy)
        #crf_time = train_crf(trainx, trainy_time)
        #crf_event = train_crf(trainx, trainy_event)

        # Test CRF
        y_pred = model.predict(testx)
        #y_pred_time = crf_time.predict(testx)
        #y_pred_event = crf_event.predict(testx)

    elif modelname == "nn":
        model, encoder_model, decoder_model, dim = train_seq2seq(trainx, trainy)
        y_pred = predict_seqs(encoder_model, decoder_model, testx, dim)
        testy_labels = []
        for seq in testy:
            testy_labels.append(decode_labels(seq))
        testy = testy_labels

        # Separate time and event models
        #time_model, time_encoder_model, time_decoder_model = train_seq2seq(trainx, trainy_time)
        #y_pred_time = predict_seqs(time_encoder_model, time_decoder_model, testx)
        #testy_labels_time = []
        #for seq in testy_time:
        #    testy_labels_time.append(decode_labels(seq))
        #testy_time = testy_labels_time

        #event_model, event_encoder_model, event_decoder_model = train_seq2seq(trainx, trainy_event)
        #y_pred_event = predict_seqs(event_encoder_model, event_decoder_model, testx)
        #testy_labels_event = []
        #for seq in testy_event:
        #    testy_labels_event.append(decode_labels(seq))
        #testy_event = testy_labels_event

    # Print metrics
    print "testy: " + str(testy[0])
    print "y_pred: " + str(y_pred[0])
    #print "labels: " + str(labels[0])
    f1_score = score(testy, y_pred, model, modelname)
    #f1_score_time = score(testy_time, y_pred_time, time_model, modelname)
    #f1_score_event = score(testy_event, y_pred_event, event_model, modelname)

    # Convert the labels back to text
    #if modelname == "nn":
    #    temp_pred = []
    #    for y in y_pred:
    #        y_fixed = decode_labels(y)
    #        temp_pred.append(y_fixed)
    #    y_pred = temp_pred

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

def train_seq2seq(trainx, trainy, vec_labels=False):
    trainx = numpy.array(trainx)
    print "trainx shape: " + str(trainx.shape)
    trainy = numpy.array(trainy)
    print "trainy shape: " + str(trainy.shape)
    input_dim = trainx.shape[-1]
    output_dim = trainy.shape[-1]
    input_seq_len = trainx.shape[1]
    output_seq_len = trainy.shape[1]

    # Create decoder target data
    trainy_target = []
    zero_lab = data_util.zero_vec(output_dim)
    if not vec_labels:
        zero_lab = encode_labels([['O']])[0][0]
    #print "zero_lab shape: " + str(numpy.array(zero_lab))
    for i in range(trainy.shape[0]):
        row = trainy[i].tolist()
        new_row = row[1:]
        new_row.append(zero_lab)
        trainy_target.append(new_row)
    trainy_target = numpy.array(trainy_target)

    print "trainy_target shape: " + str(trainy_target.shape)
        
    # Set up the encoder
    latent_dim = 100
    dropout = 0.1
    encoder_inputs = Input(shape=(None, input_dim)) #seq_len
    encoder = LSTM(latent_dim, return_state=True)

    # Encoder-Decoder model
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, output_dim))
    decoder_rnn = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, d_state_h, d_state_c = decoder_rnn(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(output_dim, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([trainx, trainy], trainy_target, epochs=10)

    # Normal RNN
    #rnn_out = GRU(latent_dim, return_sequences=False)(encoder_inputs)
    #dropout_out = Dropout(dropout)(rnn_out)
    #prediction = Dense(output_dim, activation='softmax')(dropout_out)
    #model = Model(inputs=encoder_inputs, outputs=prediction)
    #model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    #model.fit(trainx, trainy, nb_epoch=20)
    
    model.summary()                                       
    model.save('seq2seq.model')

    # Create models for inference
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_rnn(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)

    return model, encoder_model, decoder_model, output_dim

def decode_sequence(encoder_model, decoder_model, input_seq, output_seq_len, output_dim, vec_labels=False):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq, batch_size=1)

    # Generate empty target sequence of length 1.
    #output_dim = 5
    #print "output_dim: " + str(output_dim)
    target_seq = numpy.zeros((1, 1, int(output_dim)))
    # Populate the first character of target sequence with the start character.
    zero_lab = data_util.zero_vec(output_dim)
    if vec_labels:
        target_seq[0, 0] = zero_lab
    else:
        zero_lab = encode_labels([['O']])[0][0]
        index = zero_lab.index(1)
        target_seq[0, 0, index] = 1

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = []
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        #sampled_token_index = np.argmax(output_tokens[0, -1, :])
        #sampled_lab = reverse_target_char_index[sampled_token_index]
        #print "output_tokens shape: " + str(output_tokens.shape)
        token = output_tokens[0, -1]
        #print "token: " + str(token)
        encoded_label = numpy.zeros((output_dim,), dtype=numpy.int).tolist()
        if vec_labels:
            decoded_sentence.append(encoded_label)
        else:
            ind = numpy.argmax(token)
            encoded_label[ind] = 1
            #print "encoded_label: " + str(encoded_label)
            sampled_lab = decode_labels([encoded_label])[0]
            #print "sampled_lab: " + str(sampled_lab)
            decoded_sentence.append(sampled_lab)

        # Exit condition: either hit max length or find stop character.
        if (len(decoded_sentence) > output_seq_len):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = numpy.zeros((1, 1, output_dim)) 
        for x in range(output_dim):
            target_seq[0, 0, x] = token[x]

        # Update states
        states_value = [h, c]

    return decoded_sentence

''' Predict sequences for test input
    encoder_model: the encoder model
    decoder_model: the decoder model
    testx: the test input: [num_samples, max_seq_len, output_dim)
'''
def predict_seqs(encoder_model, decoder_model, testx, output_seq_len, output_dim, vec_labels=False):
    testy_pred = []
    print "output_seq_len: " + str(output_seq_len)
    print "output_dim: " + str(output_dim)
    print "vec_labels: " + str(vec_labels)
    for test_seq in testx:
        input_seq = []
        input_seq.append(test_seq)
        input_seq = numpy.array(input_seq)
        #print "input_seq shape: " + str(input_seq.shape)
        decoded_sentence = decode_sequence(encoder_model, decoder_model, input_seq, output_seq_len, output_dim, vec_labels)
        #print('-')
        #print('Input seq:', test_seq)
        #print('Predicted:', decoded_sentence)
        testy_pred.append(decoded_sentence)
    return testy_pred

def score(testy, y_pred, model, modelname):
    # Ignore O tags for evaluation
    if modelname == "crf":
        labels = list(model.classes_)
    elif modelname == "nn":
        labels = list(label_set)
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
def get_feats(seqs, train=False):
    print "get_feats"
    vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
    word2vec, dim = data_util.load_word2vec(vecfile)
    zero_vec = data_util.zero_vec(dim)
    feats = []
    labels = []
    global label_set
    label_set = set([])
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
            label_set.add(pair[1])
        feats.append(s_feats)
        labels.append(s_labels)
    if train:
        num_labels = len(list(label_set))
        create_labelencoder(list(label_set), num_labels)
        global max_seq_len
        max_seq_len = max([len(txt) for txt in feats])
        print "max_seq_len: " + str(max_seq_len)

    # Pad sequences
    #feats = pad_sequences(numpy.array(feats), maxlen=max_seq_len, dtype='float32', padding="pre")
    #labels = pad_sequences(numpy.array(labels), maxlen=max_seq_len, dtype='str', padding="pre", value='O')

    padded_feats = []
    padded_labels = []
    for feat in feats:
        pad_size = max_seq_len - len(feat)
        new_feat = []
        new_feat.append(zero_vec)
        for w in feat:
            new_feat.append(w)
        for k in range(pad_size):
            new_feat.append(zero_vec)
        padded_feats.append(new_feat)
    for labs in labels:
        pad_size = max_seq_len - len(labs)
        new_labs = []
        new_labs.append('O')
        for w in labs:
            new_labs.append(w)
        for k in range(pad_size):
            new_labs.append('O')
        padded_labels.append(new_labs)
    feats = padded_feats
    labels = padded_labels
    
    # Encode labels
    encoded_labels = encode_labels(labels)
    #for row in labels:
    #    encoded_row = encode_labels(row)
    #    encoded_labels.append(encoded_row)
    print "feats: " + str(len(feats)) + " labels: " + str(len(encoded_labels))
    return feats, encoded_labels
                          
def get_seqs(filename, split_sents=False):
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
        if split_sents:
            sents = narr.split('.')
            for sent in sents:
                sent_seq = xmltoseq.xml_to_seq(sent.strip())
                seqs.append(sent_seq)
        else:
            narr_seq = xmltoseq.xml_to_seq(narr)
            seqs.append(narr_seq)

    return ids, seqs

def create_labelencoder(data, num=0):
    global labelencoder, onehotencoder, num_labels
    labelencoder = LabelEncoder()
    labelencoder.fit(data)
    num_labels = len(labelencoder.classes_)
    #onehotencoder = OneHotEncoder()
    #onehotencoder.fit(data2)

    return labelencoder

''' Encodes labels as one-hot vectors (entire dataset: 2D array)
    data: a 1D array of labels
    num_labels: the number of label classes
'''
def encode_labels(data, labenc=None):
    if labenc == None:
        labenc = labelencoder
    if labenc == None: # or onehotencoder == None:
        print "Error: labelencoder must be trained before it can be used!"
        return None
    #return onehotencoder.transform(labelencoder.transform(data))
    data2 = []
    num_labels = len(labenc.classes_)
    print "data: " + str(len(data))
    for item in data:
        #print "item: " + str(len(item))
        item2 = labenc.transform(item)
        new_item = []
        for lab in item2:
            onehot = []
            for x in range(num_labels):
                onehot.append(0)
            onehot[lab] = 1
            new_item.append(onehot)
        data2.append(new_item)
    return data2

''' Decodes one sequence of labels
'''
def decode_labels(data, labenc=None):
    print "decode_labels"
    if labenc is None:
        labenc = labelencoder
    data2 = []
    for row in data:
        #print "- row: " + str(row)
        lab = row.index(1)
        #print "- lab: " + str(lab)
        data2.append(lab)
    #print "- data2: " + str(data2)
    return labenc.inverse_transform(data2)
    #return labelencoder.inverse_transform(onehotencoder.reverse_transform(data))

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
