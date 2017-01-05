#!/usr/bin/python
# -*- coding: utf-8 -*-
# Extract features from the xml data
# Feature names: keyword_bow, keyword_tfidf, narr_bow, narr_tfidf

from lxml import etree
import sklearn.feature_extraction
import argparse
import os
import string
import time

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--trainfeats', action="store", dest="trainoutfile")
    argparser.add_argument('--testfeats', action="store", dest="testoutfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--features', action="store", dest="featurenames")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile and args.trainoutfile and args.testoutfile):
        print "usage: ./extract_features.py --train [file.xml] --test [file.xml] --trainfeats [train.feats] --testfeats [test.feats] --labels [labelname] --features [f1,f2,f3]"
        print "labels: Final_code, ICD_cat"
        print "features: checklist, kw_bow, kw_tfidf, kw_phrase, kw_count, narr_bow, narr_count, narr_tfidf, narr_ngram, narr_vec"
        exit()

    if args.labelname:
        run(args.trainfile, args.trainoutfile, args.testfile, args.testoutfile, args.featurenames, args.labelname)
    else:
        run(args.trainfile, args.trainoutfile, args.testfile, args.testoutfile, args.featurenames)

def run(arg_train_in, arg_train_out, arg_test_in, arg_test_out, arg_featurenames="narr_count", arg_labelname="Final_Code"):
    print "extract_features"

    # Timing
    starttime = time.time()

    global vecfile, labelname
    vecfile = "/u/sjeblee/research/va/data/datasets/narratives.vectors"
    labelname = arg_labelname

    global featurenames, rec_type, checklist, dem, kw_bow, kw_tfidf, narr_bow, kw_count, narr_count, narr_tfidf, narr_vec
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
    featurenames = arg_featurenames.split(',')
    print "Features: " + str(featurenames)

    # Ngram feature params
    global min_ngram, max_ngram
    min_ngram = 1
    max_ngram = 1

    global translate_table
    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), None) for char in not_letters_or_digits)

    global kw_features, narr_features, stopwords
    kw_features = kw_bow in featurenames or kw_tfidf in featurenames or kw_phrase in featurenames or kw_count in featurenames
    narr_features = narr_bow in featurenames or narr_tfidf in featurenames or narr_count in featurenames or narr_vec in featurenames
    print "narr_features: " + str(narr_features)
    stopwords = []

    if kw_features or narr_features:
        with open("stopwords_small.txt", "r") as f:
            for line in f:
                stopwords.append(line.strip())                                    

    keys = extract(arg_train_in, arg_train_out, None)
    extract(arg_test_in, arg_test_out, keys)

    endtime = time.time()
    totaltime = endtime - starttime
    print "feature extraction took " + str(totaltime/60) + " mins"

def extract(infile, outfile, dict_keys):
    train = False
    narratives = []
    keywords = []
    
    # Get the xml from file
    root = etree.parse(infile).getroot()

    if dict_keys == None:
        train = True

        # Set up the keys for the feature vector
        dict_keys = ["MG_ID", labelname]
        if checklist in featurenames:
            dict_keys = dict_keys + ["CL_DeathAge", "CL_ageunit", "CL_DeceasedSex", "CL_Occupation", "CL_Marital", "CL_Hypertension", "CL_Heart", "CL_Stroke", "CL_Diabetes", "CL_TB", "CL_HIV", "CL_Cancer", "CL_Asthma","CL_InjuryHistory", "CL_SmokeD", "CL_AlcoholD", "CL_ApplytobaccoD"]
        elif dem in featurenames:
            dict_keys = dict_keys + ["CL_DeathAge", "CL_DeceasedSex"]
        print "dict_keys: " + str(dict_keys)
        #keywords = set([])
        #narrwords = set([])

    # Extract features
    matrix = []
    for child in root:
        features = {}

        if rec_type in featurenames:
            features["CL_" + rec_type] = child.tag

        # CHECKLIST features
        for key in dict_keys:
            #print "- key: " + key
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
        if narr_features:
            narr_string = ""
            item = child.find("narrative")
            if item != None:
                narr_string = item.text.encode("utf-8")
            narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')]
            narratives.append(narr_string.lower())

        # Save features
        matrix.append(features)

    # Construct the feature matrix

    # COUNT or TFIDF features
    if narr_count in featurenames or kw_count in featurenames or narr_tfidf in featurenames or kw_tfidf in featurenames:
        documents = []
        if narr_count in featurenames:
            documents = narratives
            print "narratives: " + str(len(narratives))
        elif kw_count in featurenames:
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

            tfidfTransformer = sklearn.feature_extraction.text.TfidfTransformer()

            # Use the training count matrix for fitting
            if train:
                tfidfTransformer.fit(count_matrix)

            # Convert matrix to tfidf
            tfidf_matrix = tfidfTransformer.transform(count_matrix)
            print "count_matrix: " + str(len(count_matrix))
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

    # WORD2VEC features
    elif narr_vec in featurenames:
        print "Warning: using word2vec features, ignoring all other features"

        # Create word2vec mapping
        word2vec = {}
        with open(vecfile, "r") as f:
            firstline = True
            for line in f:
                # Ignore the first line of the file
                if firstline:
                    firstline = False
                else:
                    #print "line: " + line
                    tokens = line.strip().split(' ')
                    vec = []
                    word = tokens[0]
                    for token in tokens[1:]:
                        #print "token: " + token
                        vec.append(float(token))
                    word2vec[word] = vec

        # Convert words to vectors and add to matrix
        dict_keys.append(narr_vec)
        maxlen = 0
        zero_vec = []
        for z in range(0, 200):
            zero_vec.append(0)
        for x in range(len(matrix)):
            narr = narratives[x]
            #print "narr: " + narr
            vectors = []
            for word in narr.split(' '):
                if len(word) > 0:
                    #if word == "didnt":
                    #    word = "didn't"
                    vec = word2vec[word]
                    vectors.append(vec)
            length = len(vectors)
            if length > maxlen:
                maxlen = length
            (matrix[x])[narr_vec] = vectors

        # Pad the narr_vecs with 0 vectors
        print "padding vectors to reach maxlen " + str(maxlen)
        for x in range(len(matrix)):
            length = len(matrix[x][narr_vec])
            if length < maxlen:
                for k in range(0, maxlen-length):
                    matrix[x][narr_vec].append(zero_vec)
                

    # Write the features to file
    print "writing " + str(len(matrix)) + " feature vectors to file..."
    output = open(outfile, 'w')
    for feat in matrix:
        #print "ICD_cat: " + feat["ICD_cat"]
        output.write(str(feat) + "\n")
    output.close()

    key_output = open(outfile + ".keys", "w")
    key_output.write(str(dict_keys))
    key_output.close() 

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

if __name__ == "__main__":main()
