#!/usr/bin/python
# -*- coding: utf-8 -*-
# Extract features from the xml data
# Feature names: keyword_bow, keyword_tfidf, narr_bow, narr_tfidf

from lxml import etree
from sklearn.decomposition import LatentDirichletAllocation
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import hashing_trick
import sklearn.feature_extraction
import argparse
import numpy
import os
import string
import time

import data_util
import preprocessing
import rebalance

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--trainfeats', action="store", dest="trainoutfile")
    argparser.add_argument('--testfeats', action="store", dest="testoutfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--features', action="store", dest="featurenames")
    argparser.add_argument('--element', action="store", dest="element")
    argparser.add_argument('--rebalance', action="store", dest="rebalance")
    argparser.add_argument('-v', '--vecfile', action="store", dest="vecfile")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile and args.trainoutfile and args.testoutfile):
        print "usage: ./extract_features.py --train [file.xml] --test [file.xml] --trainfeats [train.feats] --testfeats [test.feats] --labels [labelname] --features [f1,f2,f3] --rebalance [adasyn/smote]"
        print "labels: Final_code, ICD_cat"
        print "features: checklist, kw_bow, kw_tfidf, kw_phrase, kw_count, lda, narr_bow, narr_count, narr_tfidf, narr_ngram, narr_vec, narr_seq, event_vec, event_seq"
        exit()

    vecfile = ""
    if args.vecfile:
        vecfile = args.vecfile

    if args.labelname:
        run(args.trainfile, args.trainoutfile, args.testfile, args.testoutfile, args.featurenames, args.labelname, arg_vecfile=vecfile)
    else:
        run(args.trainfile, args.trainoutfile, args.testfile, args.testoutfile, args.featurenames, arg_vecfile=vecfile)

def run(arg_train_in, arg_train_out, arg_test_in, arg_test_out, arg_featurenames="narr_count", arg_labelname="Final_Code", stem=False, lemma=False, arg_element="narrative", arg_vecfile=""):
    print "extract_features from " + arg_train_in + " and " + arg_test_in + " : " + arg_element

    # Timing
    starttime = time.time()

    global vecfile, labelname
    if arg_vecfile == "":
        vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
    else:
        vecfile = arg_vecfile
    labelname = arg_labelname

    global featurenames, rec_type, checklist, dem, kw_bow, kw_tfidf, narr_bow, kw_count, narr_count, narr_tfidf, narr_vec, narr_seq, event_vec, event_seq, lda, symp_train
    rec_type = "type"
    checklist = "checklist"
    dem = "dem"
    kw_phrase = "kw_phrase"
    kw_bow = "kw_bow"
    kw_count = "kw_count"
    kw_tfidf ="kw_tfidf"
    narr_bow = "narr_bow"
    narr_count = "narr_count"
    narr_tfidf = "narr_tfidf"
    narr_vec = "narr_vec"
    narr_seq = "narr_seq"
    event_vec = "event_vec"
    event_seq = "event_seq"
    lda = "lda"
    symp_train = "symp_train"
    featurenames = arg_featurenames.split(',')
    print "Features: " + str(featurenames)

    # Ngram feature params
    global min_ngram, max_ngram
    min_ngram = 1
    max_ngram = 1

    # LDA feature params
    global num_topics
    num_topics = 200

    global translate_table
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), None) for char in not_letters_or_digits)

    global kw_features, narr_features, stopwords
    kw_features = kw_bow in featurenames or kw_tfidf in featurenames or kw_phrase in featurenames or kw_count in featurenames
    narr_features = narr_bow in featurenames or narr_tfidf in featurenames or narr_count in featurenames or narr_vec in featurenames or narr_seq in featurenames or lda in featurenames or event_vec in featurenames or event_seq in featurenames
    print "narr_features: " + str(narr_features)
    stopwords = []

    if kw_features or narr_features:
        with open("stopwords_small.txt", "r") as f:
            for line in f:
                stopwords.append(line.strip())                                    

    global tfidfVectorizer
    global ldaModel

    keys = extract(arg_train_in, arg_train_out, None, stem, lemma, arg_element, vecfile)
    print "dict_keys: " + str(keys)
    extract(arg_test_in, arg_test_out, keys, stem, lemma, arg_element, arg_vecfile=vecfile)

    endtime = time.time()
    totaltime = endtime - starttime
    print "feature extraction took " + str(totaltime/60) + " mins"

def extract(infile, outfile, dict_keys, stem=False, lemma=False, element="narrative", arg_rebalance="", arg_vecfile=""):
    train = False
    narratives = []
    keywords = []

    if event_vec in featurenames or event_seq in featurenames:
        element = "narr_symp"
    
    # Get the xml from file
    root = etree.parse(infile).getroot()

    if dict_keys == None:
        train = True

        # Set up the keys for the feature vector
        dict_keys = ["MG_ID"]
        if "codex" in labelname:
            dict_keys.append("WB10_codex")
            dict_keys.append("WB10_codex2")
            dict_keys.append("WB10_codex4")
        else:
            dict_keys.append(labelname)
        if rec_type in featurenames:
            dict_keys.append("CL_" + rec_type)
        if checklist in featurenames:
            dict_keys = dict_keys + ["CL_DeathAge", "CL_ageunit", "CL_DeceasedSex", "CL_Occupation", "CL_Marital", "CL_Hypertension", "CL_Heart", "CL_Stroke", "CL_Diabetes", "CL_TB", "CL_HIV", "CL_Cancer", "CL_Asthma","CL_InjuryHistory", "CL_SmokeD", "CL_AlcoholD", "CL_ApplytobaccoD"]
        elif dem in featurenames:
            dict_keys = dict_keys + ["CL_DeathAge", "CL_DeceasedSex"]
        print "dict_keys: " + str(dict_keys)
        #keywords = set([])
        #narrwords = set([])

    print "train: " + str(train)
    print "stem: " + str(stem)
    print "lemma: " + str(lemma)
    # Extract features
    matrix = []
    for child in root:
        features = {}

        if rec_type in featurenames:
            features["CL_" + rec_type] = child.tag

        # CHECKLIST features
        for key in dict_keys:
            if key[0:3] == "CL_":
                key = key[3:]
            item = child.find(key)
            value = "0"
            if item != None:
                value = item.text
            if key == "AlcoholD" or key == "ApplytobaccoD":
                if value == 'N':
                    value = 9
            features[key] = value
            #print "-- value: " + value
            if key == "MG_ID":
                print "extracting features from: " + value

        # KEYWORD features
        if kw_features:
            keyword_string = get_keywords(child)
            # Remove punctuation and trailing spaces from keywords
            words = [s.strip().translate(string.maketrans("",""), string.punctuation) for s in keyword_string.split(',')]
            # Split keyword phrases into individual words
            for word in words:
                w = word.split(' ')
                words.remove(word)
                for wx in w:
                    words.append(wx.strip().strip('–'))
            keywords.append(" ".join(words))

        # NARRATIVE features
        if narr_features or ((not train) and (symp_train in featurenames)):
            narr_string = ""
            item = child.find(element)
            if item != None:
                narr_string = data_util.stringify_children(item).encode('utf-8')
                
                if narr_string == "":
                    print "warning: empty narrative"
                narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')]
                text = " ".join(narr_words)

                if stem:
                    narr_string = preprocessing.stem(text)
                elif lemma:
                    narr_string = preprocessing.lemmatize(text)
            narratives.append(narr_string.strip().lower())
            #print "Adding narr: " + narr_string.lower()

        # SYMPTOM features
        elif train and (symp_train in featurenames):
            narr_string = ""
            item = child.find("narrative_symptoms")
            if item != None:
                item_text = item.text
                if item_text != None and len(item_text) > 0:
                    narr_string = item.text.encode("utf-8")
                    #narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')]
            narratives.append(narr_string.lower())
            print "Adding symp_narr: " + narr_string.lower()

        # Save features
        matrix.append(features)

    # Construct the feature matrix

    # COUNT or TFIDF features
    if narr_count in featurenames or kw_count in featurenames or narr_tfidf in featurenames or kw_tfidf in featurenames or lda in featurenames or symp_train in featurenames:
        documents = []
        if narr_count in featurenames or narr_tfidf in featurenames or lda in featurenames or symp_train in featurenames:
            documents = narratives
            print "narratives: " + str(len(narratives))
        elif kw_count in featurenames or kw_tfidf in featurenames:
            documents = keywords
            print "keywords: " + str(len(keywords))

        # Create count matrix
        global count_vectorizer
        if train:
            print "training count_vectorizer"
            count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(min_ngram,max_ngram),stop_words=stopwords)
            count_vectorizer.fit(documents)
            dict_keys = dict_keys + count_vectorizer.get_feature_names()
        print "transforming data with count_vectorizer"
        count_matrix = count_vectorizer.transform(documents)
        matrix_keys = count_vectorizer.get_feature_names()

        print "writing count matrix to file"
        out_matrix = open(infile + ".countmatrix", "w")
        out_matrix.write(str(count_matrix))
        out_matrix.close()

        # Add count features to the dictionary
        for x in range(len(matrix)):
            feat = matrix[x]
            for i in range(len(matrix_keys)):
                key = matrix_keys[i]
                val = count_matrix[x,i]
                feat[key] = val

        # Convert counts to TFIDF
        if (narr_tfidf in featurenames) or (kw_tfidf in featurenames):
            print "converting to tfidf..."
            print "matrix_keys: " + str(len(matrix_keys))

            # Use the training count matrix for fitting
            if train:
                global tfidfTransformer
                tfidfTransformer = sklearn.feature_extraction.text.TfidfTransformer()
                tfidfTransformer.fit(count_matrix)

            # Convert matrix to tfidf
            tfidf_matrix = tfidfTransformer.transform(count_matrix)
            print "count_matrix: " + str(count_matrix.shape)
            print "tfidf_matrix: " + str(tfidf_matrix.shape)

            # Replace features in matrix with tfidf
            for x in range(len(matrix)):
                feat = matrix[x]
                #values = tfidf_matrix[x,0:]
                #print "values: " + str(values.shape[0])
                for i in range(len(matrix_keys)):
                    key = matrix_keys[i]
                    val = tfidf_matrix[x,i]
                    feat[key] = val

        # LDA topic modeling features
        if lda in featurenames:
            global ldaModel
            if train:
                ldaModel = LatentDirichletAllocation(n_topics=num_topics)
                ldaModel.fit(count_matrix)
            lda_matrix = ldaModel.transform(count_matrix)
            for t in range(0,num_topics):
                dict_keys.append("lda_topic_" + str(t))
            for x in range(len(matrix)):
                for y in range(len(lda_matrix[x])):
                    val = lda_matrix[x][y]
                    matrix[x]["lda_topic_" + str(y)] = val

            # TODO: Print LDA topics

    # WORD2VEC features
    elif narr_vec in featurenames or event_vec in featurenames or event_seq in featurenames:
        feat_name = narr_vec
        if event_vec in featurenames:
            feat_name = event_vec
        elif event_seq in featurenames:
            feat_name = event_seq

        matrix, dict_keys = vector_features(feat_name, narratives, matrix, dict_keys, vecfile)

    # narr_seq for RNN
    elif narr_seq in featurenames:
        global vocab_size, max_seq_len
        if train:
            dict_keys.append(narr_seq)
            dict_keys.append('vocab_size')
            dict_keys.append('max_seq_len')
            vocab = set()
            for narr in narratives:
                words = narr.split(' ')
                for word in words:
                    vocab.add(word)
            vocab_size = len(vocab)
            max_seq_len = 0

        sequences = []

        # Convert text into integer sequences
        for x in range(len(matrix)):
            narr = narratives[x]
            seq = hashing_trick(narr, vocab_size, hash_function='md5', filters='\t\n', lower=True, split=' ')
            if len(seq) > max_seq_len:
                max_seq_len = len(seq)
            sequences.append(seq)

        # Pad the sequences
        sequences = pad_sequences(sequences, maxlen=max_seq_len, dtype='int32', padding='pre')
        for x in range(len(matrix)):
            matrix[x]['narr_seq'] = sequences[x]
            matrix[x]['vocab_size'] = vocab_size
            matrix[x]['max_seq_len'] = max_seq_len

    #if arg_rebalance != "":
    #    matrix_re = rebalance_data(matrix, dict_keys, arg_rebalance)
    #    write_to_file(matrix_re, dict_keys, outfile)
    #else:
    data_util.write_to_file(matrix, dict_keys, outfile)

def vector_features(feat_name, narratives, matrix, dict_keys, vecfile):
    word2vec, dim = load_word2vec(vecfile)
    global max_seq_len
    max_seq_len = 50
    dict_keys.append(feat_name)
    # Convert words to vectors and add to matrix
    if feat_name == narr_vec:
        max_seq_len = 200
    #if train:
    #max_seq_len = 0
    print "word2vec dim: " + str(dim)
    print "initial max_seq_len: " + str(max_seq_len)
    print "narratives: " + str(len(narratives))
    zero_vec = []
    for z in range(0, dim):
        zero_vec.append(0)
    for x in range(len(narratives)):
        narr = narratives[x]
        #print "narr: " + narr
        vectors = []
        tags = ['EVENT']#, 'TIMEX3']

        if feat_name == event_vec or feat_name == narr_vec:
            if feat_name == event_vec:
                narr = data_util.text_from_tags(narr, tags)
            #print "narr_filtered: " + narr
            vec = zero_vec
            for word in narr.split(' '):
                if len(word) > 0:
                    #if word == "didnt":
                    #    word = "didn't"
                    if word in word2vec:
                        vec = word2vec[word]
                    else:
                        vec = zero_vec
                    vectors.append(vec)
        elif feat_name == event_seq:
            phrases = data_util.phrases_from_tags(narr, tags)
            print "phrases: " + str(len(phrases))
            # TODO: what if phrases is empty???
            for phrase in phrases:
                word_vecs = []
                for word in phrase['text'].split(' '):
                    if word in word2vec:
                        vec = word2vec[word]
                    else:
                        vec = zero_vec
                    word_vecs.append(vec)
                if len(word_vecs) > 1:
                    vector = numpy.average(numpy.asarray(word_vecs), axis=0)
                else:
                    vector = word_vecs[0]
                #print "vector: " + str(vector)
                vectors.append(vector)
        length = len(vectors)
        if length > max_seq_len:
            # Uncomment for dynamic max_seq_len
            #if train:
            #    max_seq_len = length
            vectors = vectors[(-1*max_seq_len):]
        (matrix[x])[feat_name] = vectors

    # TODO: move this to a function
    # Pad the narr_vecs with 0 vectors
    print "padding vectors to reach maxlen " + str(max_seq_len)
    for x in range(len(matrix)):
        length = len(matrix[x][feat_name])
        matrix[x]['max_seq_len'] = max_seq_len
        if length < max_seq_len:
            for k in range(0, max_seq_len-length):
                matrix[x][feat_name].insert(0,zero_vec) # use insert for pre-padding

    return matrix, dict_keys

def rebalance_data(matrix, dict_keys, rebal_name):
    # Construct feature vectors and label vector
    features = []
    labels = []
    keys = []
    sizes = []
    flag = True
    for row in matrix:
        feats = []
        for key in dict_keys:
            val = row[key]
            if key == labelname:
                labels.append(val)
            elif key != "MG_ID":
                if type(val) is list:
                    feats = feats + val
                    if flag:
                        keys.append(key)
                        sizes.append(len(val))
                else:
                    feats.append(val)
                    if flag:
                        keys.append(key)
                        sizes.append(1)
        features.append(feats)
        if flag:
            flag = False
    new_feats, new_labels = rebalance.rebalance(features, labels, rebal_name)
    new_matrix = []
    for y in range(len(new_feats)):
        row = new_feats[y]
        label = labels[y]
        index = 0
        new_entry = {}
        for x in range(len(sizes)):
            key = keys[x]
            size = sizes[x]
            val = row[index:index+size]
            index = index+size
            new_entry[key] = val
        new_entry[labelname] = label
        new_matrix.append(new_entry)
    return new_matrix

def get_keywords(elem):
    keyword_string = ""
    keywords1 = elem.find('CODINGKEYWORDS1')
    if keywords1 != None:
        keyword_string = keyword_string + keywords1.text.encode("utf-8")
    keywords2 = elem.find('CODINGKEYWORDS2')
    if keywords2 != None:
        if keyword_string != "":
            keyword_string = keyword_string + ","
        keyword_string = keyword_string + keywords2.text.encode("utf-8")
    return keyword_string.lower()

def add_keywords(keywords, keyword_string, translate_table, stopwords):
    for word in keyword_string.split(','):
        word = word.translate(string.maketrans("",""), string.punctuation)
        w0 = word.strip()
        if "kw_phrase" in featurenames:
            w1 = ""
            for w in w0.split(' '):
                w2 = w.strip().strip('-')
                if w2 not in stopwords:
                    w1 = w1 + " " + w2
            keywords.add(w1.strip())
        else:
            for w in w0.split(' '):
                w2 = w.strip().strip('-')
                if w2 not in stopwords:
                    keywords.add(w2)

def get_narrs(filename, element="narrative"):
    narratives = []
    # Get the xml from file
    root = etree.parse(filename).getroot()
    for child in root:
        item = child.find(element)
        if item != None:
            narr_string = ""
            if item.text != None:
                narr_string = item.text.encode("utf-8")
            else:
                print "warning: empty narrative"
            # Uncomment for removing punctuation
            #narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')]
            #narr_string = " ".join(narr_words)
            print narr_string.strip().lower()
            narratives.append(narr_string.strip().lower())
    return narratives

def load_word2vec(vecfile):
    # Create word2vec mapping
    word2vec = {}
    dim = 0
    with open(vecfile, "r") as f:
        firstline = True
        for line in f:
            # Ignore the first line of the file
            if firstline:
                firstline = False
            else:
                tokens = line.strip().split(' ')
                vec = []
                word = tokens[0]
                for token in tokens[1:]:
                    vec.append(float(token))
                word2vec[word] = vec
                dim = len(vec)
    return word2vec, dim

if __name__ == "__main__":main()
