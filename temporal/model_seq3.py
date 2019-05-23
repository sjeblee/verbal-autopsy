#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('..') # verbal-autopsy dir
import data_util3 as data_util
import models_pytorch
import tools
import word2vec3 as word2vec
import xmltoseq3 as xmltoseq
import xml_to_ncrf

from keras.models import Model
#from keras.layers import Input, GRU, LSTM, Dense, Dropout
#from keras.preprocessing.sequence import pad_sequences
from lxml import etree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn_crfsuite import CRF, metrics
import argparse
import numpy
import subprocess
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

global labelencoder, onehotencoder, label_set, max_seq_len, num_labels
debug = True
id_name = "record_id"
max_seq_len = 100
use_ncrf = False
#vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
#vecfile = "/u/sjeblee/research/vectors/GoogleNews-vectors-negative300.bin"
vecfile = "/u/sjeblee/research/vectors/wikipedia-pubmed-and-PMC-w2v.bin"


def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--dev', action="store", dest="devfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--model', action="store", dest="model")
    argparser.add_argument('--inline', action="store_true", dest="inline") # Use inline for TempEval dataset
    argparser.set_defaults(inline=False)
    argparser.set_defaults(devfile="")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile):
        print("usage: ./model_seq.py --train [file.xml] --test [file.xml] (--out [file.xml] --dev [dev.xml] --model [crf/gru/ncrf] --inline)")
        print("For NCRF model, use --dev path/to/dev.xml")
        print("For TempEval data, use --inline")
        exit()

    if args.outfile and args.model:
        run(args.trainfile, args.testfile, args.outfile, args.model, args.inline, devfile=args.devfile)
    elif args.outfile:
        run(args.trainfile, args.testfile, args.outfile, arg_inline=args.inline, devfile=args.devfile)
    else:
        run(args.trainfile, args.testfile, arg_inline=args.inline, devfile=args.devfile)

def run(trainfile, testfile, outfile="", modelname="crf", arg_inline=False, devfile=""):
    if 'va' in trainfile:
        split_sents = True
    
    if modelname == 'ncrf':
        global use_ncrf
        use_ncrf = True
        if devfile == "":
            print("WARNING: missing devfile argument")
        if outfile == "":
            print("WARNING: missing outfile argument")
        # Convert data to ncrf format
        ncrf_path = '/u/sjeblee/research/git/NCRFpp'
        data_name = 'thyme'
        ncrf_dir = '/u/sjeblee/research/git/NCRFpp/va_data'
        if 'TempEval' in trainfile:
            ncrf_dir = '/u/sjeblee/research/git/NCRFpp/tempeval_data'
            data_name = 'tempeval'
        elif 'thyme' in trainfile:
            ncrf_dir = '/u/sjeblee/research/git/NCRFpp/thyme_data'
        train_ncrf = ncrf_dir + '/train.bio'
        dev_ncrf = ncrf_dir + '/dev.bio'
        test_ncrf = ncrf_dir + '/test.bio'
        xml_to_ncrf.extract_features(trainfile, train_ncrf, arg_inline)
        xml_to_ncrf.extract_features(devfile, dev_ncrf, arg_inline)
        xml_to_ncrf.extract_features(testfile, test_ncrf, arg_inline)
    else:
        # Extract sequences with labels
        train_ids, train_seqs = get_seqs(trainfile, split_sents=True, inline=False, add_spaces=True)
        test_ids, test_seqs = get_seqs(testfile, split_sents=True, inline=arg_inline, add_spaces=True)
        print("test_ids:", len(test_ids))

    if modelname == "crf":
        trainx = [sent2features(s) for s in train_seqs]
        trainy = [sent2labels(s) for s in train_seqs]
        testx = [sent2features(s) for s in test_seqs]
        testy = [sent2labels(s) for s in test_seqs]
    elif modelname == "nn":
        extra_seqs = get_seqs('/u/sjeblee/research/data/TimeBank/timebank_all_timeml_simple.xml', split_sents=True)
        train_seqs = extra_seqs + train_seqs
        trainx, trainy = get_feats(train_seqs, train=True)
        testx, testy = get_feats(test_seqs)
    elif modelname == 'gru':
        trainx, trainy = get_feats(train_seqs, pad=False, train=True)
        testx, testy = get_feats(test_seqs, pad=False)

    if modelname != 'ncrf':
        print("train:", str(len(trainx)), " ", str(len(trainy)))
        print("test: ", str(len(testx)), " ", str(len(testy)))

    # Split labels into time and event only
    split = False
    if split:
        trainy_time, trainy_event = split_labels(trainy)
        testy_time, testy_event = split_labels(testy)

    # Train model
    model = None
    encoder_model = None
    decoder_model = None
    y_pred = []
    if modelname == 'crf':
        model = train_crf(trainx, trainy)
        #crf_time = train_crf(trainx, trainy_time)
        #crf_event = train_crf(trainx, trainy_event)

        # Test CRF
        testy_labels = testy
        y_pred_labels = model.predict(testx)
        print("testy:", len(testy), "ypred:", len(y_pred_labels))
        #y_pred_time = crf_time.predict(testx)
        #y_pred_event = crf_event.predict(testx)

    elif modelname == 'ncrf':
        # Run NCRF model
        subprocess.call(["python", "main.py", "--config", data_name + ".train.config"], cwd=ncrf_path)
        subprocess.call(["python", "main.py", "--config", data_name + ".decode.config"], cwd=ncrf_path)
        xml_to_ncrf.ncrf_to_xml(ncrf_dir, outfile, testfile)

    elif modelname == 'gru':
        nodes = 256
        epochs = 2
        model = models_pytorch.rnn_model(trainx, trainy, num_nodes=nodes, loss_function='crossentropy', num_epochs=epochs, batch_size=1, use_prev_labels=False)
        y_pred = models_pytorch.test_rnn(model, testx, batch_size=1)
        y_pred_labels = decode_all_labels(y_pred)
        testy_labels = decode_all_labels(testy)

    elif modelname == "ende":
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
    #print "testy: " + str(testy[0])
    #print "y_pred: " + str(y_pred[0])
    #print "labels: " + str(labels[0])

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
    if modelname != 'ncrf':
        f1_score = score(testy_labels, y_pred_labels, model, modelname)
        print("F1:", str(f1_score))
        test_dict = {}
        for x in range(len(test_ids)):
            rec_id = test_ids[x]
            if debug: print("x:", str(x), "rec_id:", rec_id)
            rec_seq = list(zip((item[0] for item in test_seqs[x]), y_pred_labels[x]))
            # Concatenate all the sequences for each record
            if rec_id not in test_dict:
                test_dict[rec_id] = []
            test_dict[rec_id] = test_dict[rec_id] + rec_seq # TODO: add line breaks???
            if debug: print("rec_seq:", str(rec_seq))
        xml_tree = xmltoseq.seq_to_xml(test_dict, testfile, tag="narr_timeml_"+modelname)

        # write the xml to file
        if len(outfile) > 0:
            print("Writing test output to xml file...")
            xml_tree.write(outfile)
            subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', outfile])
            subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', outfile])

    # TODO: run appropriate evaluation script


def train_crf(trainx, trainy):
    print("training CRF...")
    crf = CRF(
            algorithm='lbfgs',
            c1=0.1,
            c2=0.1,
            max_iterations=500,
            all_possible_transitions=True
        )
    crf.fit(trainx, trainy)
    return crf


def train_seq2seq(trainx, trainy, num_nodes=100, vec_labels=False, loss_function="cosine_proximity", num_epochs=10):
    trainx = numpy.array(trainx)
    print("trainx shape: ", str(trainx.shape))
    trainy = numpy.array(trainy)
    print("trainy shape: ", str(trainy.shape))
    input_dim = trainx.shape[-1]
    output_dim = trainy.shape[-1]
    input_seq_len = trainx.shape[1]
    output_seq_len = trainy.shape[1]

    # Create decoder target data
    trainy_target = []
    zero_lab = data_util.zero_vec(output_dim)
    if not vec_labels:
        zero_lab = encode_labels([['O']])[0][0]
    print("zero_lab shape: ", str(numpy.asarray(zero_lab)))
    for i in range(trainy.shape[0]):
        row = trainy[i].tolist()
        new_row = row[1:]
        new_row.append(zero_lab)
        trainy_target.append(new_row)
    trainy_target = numpy.asarray(trainy_target)

    print("trainy_target shape: ", str(trainy_target.shape))

    # Set up the encoder
    latent_dim = num_nodes
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
    model.compile(optimizer='rmsprop', loss=loss_function)
    model.fit([trainx, trainy], trainy_target, epochs=num_epochs)

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
            print("sampled_lab: ", str(sampled_lab))
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
    print("output_seq_len: ", str(output_seq_len))
    print("output_dim: ", str(output_dim))
    print("vec_labels: ", str(vec_labels))
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
    elif modelname == "nn" or modelname == 'gru':
        labels = list(label_set)
    labels.remove('O')
    f1_score = metrics.flat_f1_score(testy, y_pred, average='weighted', labels=labels)
    print("F1: ", str(f1_score))
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
def get_feats(seqs, pad=True, train=False):
    print("get_feats")
    vec_model, dim = word2vec.load(vecfile)
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
            vector = word2vec.get(word, vec_model)
            s_feats.append(vector)
            s_labels.append(pair[1])
            label_set.add(pair[1])
        feats.append(s_feats)
        labels.append(s_labels)
    if train:
        num_labels = len(list(label_set))
        if debug: print("num original labels:", str(num_labels))
        label_list = list(label_set)
        print('label_list:', str(label_list))
        label_list.append('IE') # Add the IE label in case it wasn't in training data
        print('label_list:', str(labels))
        create_labelencoder(label_list, num_labels)
        global max_seq_len
        #max_seq_len = max([len(txt) for txt in feats])
    print("max_seq_len: ", str(max_seq_len))

    # Pad sequences
    #feats = pad_sequences(numpy.array(feats), maxlen=max_seq_len, dtype='float32', padding="pre")
    #labels = pad_sequences(numpy.array(labels), maxlen=max_seq_len, dtype='str', padding="pre", value='O')

    if pad:
        padded_feats = []
        padded_labels = []
        for feat in feats:
            #print "seq len: " + str(len(feat))
            while len(feat) > max_seq_len:
                feat_part = feat[0:max_seq_len]
                padded_feats.append(pad_feat(feat_part, max_seq_len, zero_vec))
                feat = feat[max_seq_len:]
            new_feat = pad_feat(feat, max_seq_len, zero_vec)
            padded_feats.append(new_feat)
        for labs in labels:
            while len(labs) > max_seq_len:
                labs_part = labs[0:max_seq_len]
                padded_labels.append(pad_feat(labs_part, max_seq_len, 'O'))
                labs = labs[max_seq_len:]
                #print("labs:", str(labs))
            padded_labels.append(pad_feat(labs, max_seq_len, 'O'))
        feats = padded_feats
        labels = padded_labels

    # Encode labels
    print("labels[0]: ", str(labels[0]))
    encoded_labels = encode_labels(labels, max_len=max_seq_len, pad=pad)
    print("encoded_labels[0]: ", str(encoded_labels[0]))
    #for row in labels:
    #    encoded_row = encode_labels(row)
    #    encoded_labels.append(encoded_row)
    print("feats: ", str(len(feats)), " labels: ", str(len(encoded_labels)))
    return feats, encoded_labels


def pad_feat(feat, max_seq_len, pad_item):
    pad_size = max_seq_len - len(feat)
    assert(pad_size >= 0)
    new_feat = []
    #new_feat.append(pad_item) # Start symbol for encoder-decoder
    for w in feat:
        new_feat.append(w)
    for k in range(pad_size):
        new_feat.append(pad_item)
    return new_feat


def get_seqs(filename, split_sents=False, inline=True, add_spaces=False):
    print("get_seqs ", filename)
    ids = []
    narrs = []
    anns = []
    seqs = []
    seq_ids = []

    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()

    for child in root:
        narr = ""
        rec_id = child.find(id_name).text
        ids.append(rec_id)
        # Get the narrative text
        node = child.find("narr_timeml_simple")
        if inline:
            if node is None:
                narr_node = child.find("narrative")
                if narr_node is None:
                    print("no narrative: ", data_util.stringify_children(child))
                else:
                    narr = narr_node.text
                    #print "narr: " + narr
                    narrs.append(narr)
            else:
                rec_id = child.find(id_name).text
                #print "rec_id: " + rec_id
                #narr = etree.tostring(node, encoding='utf-8').decode('utf-8')
                narr = tools.stringify_children(node)
                if add_spaces:
                    narr = narr.replace('<', ' <')
                    narr = narr.replace('>', '> ')
                    narr = narr.replace('.', ' .')
                    narr = narr.replace(',', ' ,')
                    narr = narr.replace(':', ' :')
                    narr = narr.replace('  ', ' ')
                #print("narr: ", narr)
                #ids.append(rec_id)
                narrs.append(narr)
        else: # NOT inline
            anns.append(data_util.stringify_children(node).encode('utf8'))
            narr_node = child.find("narrative")
            narrs.append(narr_node.text)

    if inline:
        #split_sents = False
        for x in range(len(narrs)):
            narr = narrs[x]
            rec_id = ids[x]
            if split_sents:
                sents = narr.split('.')
                for sent in sents:
                    sent_seq = xmltoseq.xml_to_seq(sent.strip())
                    seqs.append(sent_seq)
                    seq_ids.append(rec_id)
            else:
                narr_seq = xmltoseq.xml_to_seq(narr)
                for seq in narr_seq:
                    seqs.append(seq)
                    seq_ids.append(rec_id)
    else:
        # TEMP
        use_ncrf = False
        split_sents = True
        print("split_sents: ", str(split_sents))
        for x in range(len(narrs)):
            narr = narrs[x]
            ann = anns[x]
            rec_id = ids[x]
            ann_seqs = xmltoseq.ann_to_seq(narr, ann, split_sents, use_ncrf)
            print("seqs: ", str(len(ann_seqs)))
            for s in ann_seqs:
                seqs.append(s)
                seq_ids.append(rec_id)

    if debug: print("seqs[0]", str(seqs[0]))
    return seq_ids, seqs


def create_labelencoder(data, num=0):
    global labelencoder, onehotencoder, num_labels
    print("create_labelencoder: data[0]: ", str(data[0]))
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
def encode_labels(data, labenc=None, max_len=50, pad=True):
    if labenc is None:
        labenc = labelencoder
    if labenc is None: # or onehotencoder == None:
        print("Error: labelencoder must be trained before it can be used!")
        return None
    #return onehotencoder.transform(labelencoder.transform(data))
    data2 = []

    num_labels = len(labenc.classes_)
    zero_vec = data_util.zero_vec(num_labels)
    if debug: print("data: ", str(len(data)))
    for item in data:
        #print "item len: " + str(len(item))
        new_item = []
        if len(item) > 0:
            item2 = labenc.transform(item)
            for lab in item2:
                onehot = []
                for x in range(num_labels):
                    onehot.append(0)
                onehot[lab] = 1
                new_item.append(onehot)
        # Pad vectors
        if pad:
            if len(new_item) > max_len:
                new_item = new_item[0:max_len]
            while len(new_item) < max_len:
                new_item.append(zero_vec)
        data2.append(new_item)
    return data2


''' Decodes one sequence of labels
'''
def decode_labels(data, labenc=None):
    #print "decode_labels"
    if labenc is None:
        labenc = labelencoder
    data2 = []
    for row in data:
        #print "- row: " + str(row)
        lab = numpy.argmax(numpy.asarray(row))
        #print "- lab: " + str(lab)
        data2.append(lab)
    #print "- data2: " + str(data2)
    return labenc.inverse_transform(data2)
    #return labelencoder.inverse_transform(onehotencoder.reverse_transform(data))


def decode_all_labels(data, labenc=None):
    decoded_labels = []
    for sequence in data:
        labs = decode_labels(sequence, labenc)
        decoded_labels.append(labs)
    return decoded_labels


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
        #postag1 = sent[i-1][1]
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
        #print("sent[i+1]", str(sent[i+1]))
        word1 = sent[i+1][0]
        #postag1 = sent[i+1][1]
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
