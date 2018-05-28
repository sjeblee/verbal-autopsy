#!/usr/bin/python
# -*- coding: utf-8 -*-
# Extract features from the xml data

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
import word2vec

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
        print "features: checklist, kw_bow, kw_tfidf, kw_phrase, kw_count, kw_vec, lda, narr_bow, narr_count, narr_tfidf, narr_ngram, narr_vec, narr_seq, event_vec, event_seq, dem, narr_dem"
        exit()

    vecfile = ""
    if args.vecfile:
        vecfile = args.vecfile

    if args.labelname:
        run(args.trainfile, args.trainoutfile, args.testfile, args.testoutfile, args.featurenames, args.labelname, arg_vecfile=vecfile)
    else:
        run(args.trainfile, args.trainoutfile, args.testfile, args.testoutfile, args.featurenames, arg_vecfile=vecfile)

''' Run feature extraction for the training and testing files
    This is the function that is called from pipeline.py
'''
def run(arg_train_in, arg_train_out, arg_test_in, arg_test_out, arg_featurenames="narr_count", arg_labelname="Final_Code", stem=False, lemma=False, arg_element="narrative", arg_vecfile=""):
   # print "extract_features from " + arg_train_in + " and " + arg_test_in + " : " + arg_element

    # Timing
    starttime = time.time()

    global vecfile, labelname
    if arg_vecfile == "":
        vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
    else:
        vecfile = arg_vecfile
    labelname = arg_labelname
    print "vecfile: " + vecfile

    # Feature types
    global featurenames, rec_type, checklist, dem, kw_words, kw_bow, kw_tfidf, narr_bow, kw_count, kw_vec, kw_clusters, narr_count, narr_tfidf, narr_vec, narr_seq, narr_dem, event_vec, event_seq, lda, symp_train, symp_count

    #Edit by Yoona
    global narr_symp, symp_vec
    narr_symp = "narr_symp"
    symp_vec = "symp_vec"

    rec_type = "type"
    checklist = "checklist"
    dem = "dem"
    kw_words = "kw_words"
    kw_phrase = "kw_phrase"
    kw_bow = "kw_bow"
    kw_count = "kw_count"
    kw_tfidf ="kw_tfidf"
    kw_vec = "kw_vec"
    kw_clusters = "kw_clusters"
    narr_bow = "narr_bow"
    narr_count = "narr_count"
    narr_tfidf = "narr_tfidf"
    narr_vec = "narr_vec"
    narr_seq = "narr_seq"
    narr_dem = "narr_dem"
    event_vec = "event_vec"
    event_seq = "event_seq"
    lda = "lda"
    symp_train = "symp_train"
    symp_count = "symp_count"

    # Add by yoona for training using "symp_narr"
    featurenames = arg_featurenames.split(',')
    print "Features: " + str(featurenames)

    # Ngram feature params
    global min_ngram, max_ngram
    min_ngram = 1
    max_ngram = 1

    # LDA feature params
    global num_topics
    num_topics = 100

    global translate_table
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), None) for char in not_letters_or_digits)

    global kw_features, narr_features, stopwords
    kw_features = kw_bow in featurenames or kw_tfidf in featurenames or kw_phrase in featurenames or kw_count in featurenames or kw_vec in featurenames or kw_words in featurenames
    narr_features = narr_bow in featurenames or narr_tfidf in featurenames or narr_count in featurenames or narr_vec in featurenames or narr_seq in featurenames or lda in featurenames or event_vec in featurenames or event_seq in featurenames or symp_count in featurenames
    print "narr_features: " + str(narr_features)
    stopwords = []

    # Remove a small set of stopwords from the narrative
    if kw_features or narr_features:
        with open("stopwords_small.txt", "r") as f:
            for line in f:
                stopwords.append(line.strip())                                    

    global tfidfVectorizer
    global lda_model

    # Extract the features
    keys = extract(arg_train_in, arg_train_out, None, stem, lemma, arg_element, arg_vecfile=vecfile)
    print "dict_keys: " + str(keys)
    extract(arg_test_in, arg_test_out, keys, stem, lemma, arg_element, arg_vecfile=vecfile)

    endtime = time.time()
    totaltime = endtime - starttime
    print "feature extraction took " + str(totaltime/60) + " mins"

''' Extract features from an xml file
'''
def extract(infile, outfile, dict_keys, stem=False, lemma=False, element="narrative", arg_rebalance="", arg_vecfile=""):
    print "arg_vecfile: " + arg_vecfile
    train = False
    narratives = []
    keywords = []

    # Edit by Yoona
    symptoms = []
    mult_features = []

    if event_vec in featurenames or event_seq in featurenames or symp_count in featurenames:
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
        elif dem in featurenames or narr_dem in featurenames:
            dict_keys = dict_keys + ["CL_DeathAge", "CL_ageunit", "CL_DeceasedSex"]
        elif kw_clusters in featurenames:
            dict_keys.append("keyword_clusters")
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
            orig_key = key
            if key[0:3] == "CL_":
                key = key[3:]
            item = child.find(key)
            value = "0"
            if item != None:
                value = item.text
            if key == "AlcoholD" or key == "ApplytobaccoD":
                if value == 'N':
                    value = 9
            features[orig_key] = value
            #if key == "MG_ID":
            #    print "extracting features from: " + value
            #print "-- key: " + orig_key + " value: " + value

        # KEYWORD features
        if kw_features:
            keyword_string = get_keywords(child, 'keywords_spell')
            # Remove punctuation and trailing spaces from keywords
            words = [s.strip().translate(string.maketrans("",""), string.punctuation) for s in keyword_string.split(',')]
            # Split keyword phrases into individual words
            for word in words:
                w = word.split(' ')
                words.remove(word)
                for wx in w:
                    words.append(wx.strip().strip('–'))
            keywords.append(" ".join(words))
            if kw_words in featurenames:
                max_seq_len = 30
                words = " ".join(words).split(' ')
                if len(words) > max_seq_len:
                    words = words[0:max_seq_len]
                while len(words) < max_seq_len:
                    words.append("0")
                features[kw_words] = words

        # NARRATIVE features
        if narr_features or ((not train) and (symp_train in featurenames)):
            narr_string = ""
            #item = child.find(element)
            item = child.find("narrative")
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

                if symp_count in featurenames:
                    tags = ['EVENT', 'TIMEX3']
                    narr_string = data_util.text_from_tags(narr_string, tags)

                # Add demographic information to the narrative text before generating narr_vec features
                if narr_dem in featurenames:
                    age = features["CL_DeathAge"]
                    # Convert ageunit and gender to words
                    ageunit = data_util.decode_ageunit(features["CL_ageunit"])
                    gender = data_util.decode_gender(features["CL_DeceasedSex"])
                    dem_prefix = " age " + str(age) + " " + ageunit + " " + str(gender) + " "
                    print "narr_dem: " + dem_prefix
                    narr_string = dem_prefix + narr_string
            narratives.append(narr_string.strip().lower())
            #print "Adding narr: " + narr_string.lower()

        # SYMPTOM features
        '''
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
        '''
        #SYMPTOM features (Edit by Yoona)
        if narr_symp in element:
            symp_string = ""
            item = child.find(narr_symp) # Hard-coded. To be fixed 
            if item != None:
                item_text = item.text
                symp_string = " " + str(item_text) + " "
            symptoms.append(symp_string.lower())
        

        # Save features
        matrix.append(features)

    # Construct the feature matrix

    # Concatenate narratives and symptoms. Put more weight on symptoms (Yoona)
    #for i in range(len(symptoms)):
	#narratives[i] = narratives[i] + symptoms[i] * 20


    # COUNT or TFIDF features
    if narr_count in featurenames or kw_count in featurenames or narr_tfidf in featurenames or kw_tfidf in featurenames or lda in featurenames or symp_train in featurenames or symp_count in featurenames:
        print "count features"
        documents = []
        if narr_count in featurenames or narr_tfidf in featurenames or lda in featurenames or symp_train in featurenames or symp_count in featurenames:
            documents = narratives
            print "narratives: " + str(len(narratives))
        elif kw_count in featurenames or kw_tfidf in featurenames:
            documents = keywords
            print "keywords: " + str(len(keywords))
        count_feats = False
        if narr_count in featurenames or narr_tfidf in featurenames or kw_count in featurenames:
            count_feats = True

        # Insert "symptoms" key and count matrix of symptoms as value into matrix (Yoona) 
        if narr_symp in element:
            global symp_count_vectorizer
            if train:
                symp_count_vectorizer =  sklearn.feature_extraction.text.CountVectorizer(ngram_range=(min_ngram,max_ngram),stop_words=stopwords)
                symp_count_vectorizer.fit(symptoms)
                #dict_keys = dict_keys + ["narr_symptoms"]

                temp_keys = symp_count_vectorizer.get_feature_names()
                symptom_keys = []
                for key in temp_keys:
                    symptom_keys.append("symp_" + key)
                print("Change symptoms key with appropriate name")

                if count_feats:
                    dict_keys = dict_keys + symptom_keys
            symp_count_matrix = symp_count_vectorizer.transform(symptoms)

            if count_feats:
                for x in range(len(matrix)):
                    feat = matrix[x]
                    for i in range(len(symptom_keys)):
                        key = symptom_keys[i]
                        val = symp_count_matrix[x, i]
                        feat[key] = val   
                print("Add symptoms into dictionary as a key")

        out_matrix = open(infile + ".symp_countmatrix", "w")
        out_matrix.write(str(symp_count_matrix))
        out_matrix.close()

        # Create count matrix
        global count_vectorizer
        if train:
            print "training count_vectorizer"
            count_vectorizer = sklearn.feature_extraction.text.CountVectorizer(ngram_range=(min_ngram,max_ngram),stop_words=stopwords)
            count_vectorizer.fit(documents)
	    
            if count_feats:
                dict_keys = dict_keys + count_vectorizer.get_feature_names()
        print "transforming data with count_vectorizer"
        count_matrix = count_vectorizer.transform(documents)
        matrix_keys = count_vectorizer.get_feature_names()

        print "writing count matrix to file"
        out_matrix = open(infile + ".countmatrix", "w")
        out_matrix.write(str(count_matrix))
        out_matrix.close()

        # Add count features to the dictionary
        if count_feats:
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
            global lda_model
            if train:
                lda_model = LatentDirichletAllocation(n_topics=num_topics)
                lda_model.fit(count_matrix)
            lda_matrix = lda_model.transform(count_matrix)
            for t in range(0,num_topics):
                dict_keys.append("lda_topic_" + str(t))
            for x in range(len(matrix)):
                for y in range(len(lda_matrix[x])):
                    val = lda_matrix[x][y]
                    matrix[x]["lda_topic_" + str(y)] = val

            # Save LDA topics
            vocab = count_vectorizer.get_feature_names()
            topic_words = {}
            for topic, comp in enumerate(lda_model.components_):
                    # for the n-dimensional array "arr":
                    # argsort() returns a ranked n-dimensional array of arr, call it "ranked_array"
                    # which contains the indices that would sort arr in a descending fashion
                    # for the ith element in ranked_array, ranked_array[i] represents the index of the
                    # element in arr that should be at the ith index in ranked_array
                    # ex. arr = [3,7,1,0,3,6]
                    # np.argsort(arr) -> [3, 2, 0, 4, 5, 1]
                    # word_idx contains the indices in "topic" of the top num_top_words most relevant
                    # to a given topic ... it is sorted ascending to begin with and then reversed (desc. now)
                    word_idx = numpy.argsort(comp)[::-1]
                    # store the words most relevant to the topic
                    topic_words[topic] = [vocab[i] for i in word_idx]
            print "writing lda topics to file"
            out_lda = open(infile + ".lda_topics", "w")
            out_lda.write(str(topic_words))
            out_lda.close()

    # WORD2VEC features
    if narr_vec in featurenames or event_vec in featurenames or event_seq in featurenames or kw_vec in featurenames:
        feat_name = narr_vec
        text = narratives
        if event_vec in featurenames:
            feat_name = event_vec
        elif event_seq in featurenames:
            feat_name = event_seq
        elif kw_vec in featurenames:
            feat_name = kw_vec
            text = keywords

        matrix, dict_keys = vector_features(feat_name, text, matrix, dict_keys, arg_vecfile)

        # WORD2VEC for narrative symptoms
        if narr_symp in element:
            feat_name = symp_vec
            text = symptoms
            matrix, dict_keys = vector_features(feat_name, text, matrix, dict_keys, arg_vecfile)

            # Concatenate narrative vectors and narrative_symptom vectors. 
            for x in range(len(matrix))
                feat = matrix[x]
                symptom_vec = feat[symp_vec]
                narrative_vec = feat[narr_vec]
                concatenated_vec = feat[narr_vec] + feat[symptom_vec]
                feat[narr_vec] = concatenated_vec


    # narr_seq for RNN
    if narr_seq in featurenames:
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
    if narr_dem in featurenames:
        # Remove dem features because we've already added them to the narrative
        dict_keys.remove("CL_DeceasedSex")
        dict_keys.remove("CL_DeathAge")
        dict_keys.remove("CL_ageunit")
    data_util.write_to_file(matrix, dict_keys, outfile)

''' Get vector features
'''
def vector_features(feat_name, narratives, matrix, dict_keys, vecfile):
    print "vecfile: " + vecfile
    vec_model, dim = word2vec.load(vecfile)
    global max_seq_len

    max_seq_len = 30
    dict_keys.append(feat_name)
    # Convert words to vectors and add to matrix
    if feat_name == narr_vec:
        max_seq_len = 200

    # Edit by Yoona. Deal with narrative symptoms
    if feat_name == symp_vec:
        max_seq_len = 100
    #if train:
    #max_seq_len = 0
    print "feat_name: " + feat_name
    print "word2vec dim: " + str(dim)
    print "initial max_seq_len: " + str(max_seq_len)
    print "narratives: " + str(len(narratives))
    zero_vec = data_util.zero_vec(dim)
    for x in range(len(narratives)):
        narr = narratives[x]
        #print "narr: " + narr
        vectors = []
        tags = ['EVENT', 'TIMEX3']

        if feat_name == event_vec or feat_name == narr_vec or feat_name == kw_vec or feat_name == symp_vec:
            if feat_name == event_vec:
                narr = data_util.text_from_tags(narr, tags)
                print "narr_filtered: " + narr
            vec = zero_vec
            for word in narr.split(' '):
                if len(word) > 0:
                    #if word == "didnt":
                    #    word = "didn't"
                    vec = word2vec.get(word, vec_model)
                    vectors.append(vec)
        elif feat_name == event_seq:
            phrases = data_util.phrases_from_tags(narr, tags)
            print "phrases: " + str(len(phrases))
            # TODO: what if phrases is empty???
            for phrase in phrases:
                word_vecs = []
                for word in phrase['text'].split(' '):
                    vec = word2vec.get(word, vec_model)
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

'''
   Get the physician-generated keyword from a record
   elem: the xml element representing the record
'''
def get_keywords(elem, name=None):
    keyword_string = ""
    if name is None:
        keywords1 = elem.find('CODINGKEYWORDS1')
        if keywords1 is not None and keywords1.text is not None:
            keyword_string = keyword_string + keywords1.text.encode("utf-8")
        keywords2 = elem.find('CODINGKEYWORDS2')
        if keywords2 is not None and keywords2.text is not None:
            if keyword_string != "":
                keyword_string = keyword_string + ","
            keyword_string = keyword_string + keywords2.text.encode("utf-8")
    else:
        keywords = elem.find(name)
        if keywords is not None and keywords.text is not None:
            keyword_string = keywords.text.encode("utf-8")

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

'''
Get the narratives from an xml file
'''
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
            narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')]
            narr_string = " ".join(narr_words)
            #print narr_string.strip().lower()
            narratives.append(narr_string.strip().lower())
    return narratives

if __name__ == "__main__":main()
