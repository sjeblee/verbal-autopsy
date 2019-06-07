#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Extract features from the xml data
# Feature names: keyword_bow, keyword_tfidf, narr_bow, narr_tfidf

from lxml import etree
import sklearn.feature_extraction
import argparse
import os
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--features', action="store", dest="featurenames")
    argparser.add_argument('--keys', action="store", dest="keyfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print('usage: ./raw2vector.py --input [file.xml] --output [outfile] --labels [labelname] --features [f1,f2,f3]')
        print('labels: Final_code, ICD_cat')
        print('features: checklist, kw_bow, kw_tfidf, kw_phrase, narr_bow, narr_tfidf')
        exit()

    labelname = 'Final_code'
    if args.labelname:
        labelname = args.labelname

    global featurenames
    rec_type = 'type'
    checklist = 'checklist'
    dem = 'dem'
    kw_phrase = 'kw_phrase'
    kw_bow = 'kw_bow'
    kw_tfidf = 'kw_tfidf'
    narr_bow = 'narr_bow'
    narr_count = 'narr_count'
    narr_tfidf = 'narr_tfidf'
    featurenames = [checklist, kw_bow]
    if args.featurenames:
        featurenames = args.featurenames.split(',')
    print('Features:', str(featurenames))

    # Ngram feature params
    global min_ngram, max_ngram
    min_ngram = 1
    max_ngram = 2

    # Read in feature keys
    global keys
    keys = []
    usekeys = False
    if args.keyfile:
        usekeys = True
        with open(args.keyfile, "r") as kfile:
            keys = eval(kfile.read())

    # Set up the keys for the feature vector
    dict_keys = ["MG_ID", labelname]
    if checklist in featurenames:
        dict_keys = dict_keys + ["CL_DeathAge", "CL_ageunit", "CL_DeceasedSex", "CL_Occupation", "CL_Marital", "CL_Hypertension", "CL_Heart", "CL_Stroke", "CL_Diabetes", "CL_TB", "CL_HIV", "CL_Cancer", "CL_Asthma","CL_InjuryHistory", "CL_SmokeD", "CL_AlcoholD", "CL_ApplytobaccoD"]
    elif dem in featurenames:
        dict_keys = dict_keys + ["CL_DeathAge", "CL_DeceasedSex"]
    keywords = set([])
    narrwords = set([])

    not_letters_or_digits = u'!"#%\'()*+,-./:;<=>?@[\]^_`{|}~'
    translate_table = dict((ord(char), None) for char in not_letters_or_digits)

    # Get the xml from file
    root = etree.parse(args.infile).getroot()

    kw_features = kw_bow in featurenames or kw_tfidf in featurenames or kw_phrase in featurenames
    narr_features = narr_bow in featurenames or narr_tfidf in featurenames or narr_count in featurenames
    stopwords = []

    if kw_features or narr_features:
        with open("stopwords_small.txt", "r") as f:
            for line in f:
                stopwords.append(line.strip())

    # KEYWORDS setup
    if kw_features:
        for child in root:
            keyword_string = get_keywords(child)
            add_keywords(keywords, keyword_string, translate_table, stopwords)
        print('Keywords:', str(len(keywords)))

    # NARRATIVE setup
    if narr_features:
        for child in root:
            narr_string = ""
            node = child.find("narrative")
            if node is not None:
                narr_string = node.text.encode("utf-8")
            words = narr_string.lower().translate(string.maketrans("", ""), string.punctuation).split(' ')
            for word in words:
                if word not in stopwords:
                    narrwords.add(word.strip())

        print('Words:', str(len(narrwords)))

    # Extract features
    matrix = []
    for child in root:
        features = {}

        if rec_type in featurenames:
            features[rec_type] = child.tag

        # CHECKLIST features
        for key in dict_keys:
            #print "- key: " + key
            if key[0:3] == "CL_":
                key = key[3:]
            item = child.find(key)
            value = "0"
            if item is not None:
                value = item.text
            if key == "AlcoholD" or key == "ApplytobaccoD":
                if value == 'N':
                    value = 9
            features[key] = value

        # KEYWORD features
        if kw_features:
            keyword_string = get_keywords(child)
            # Remove punctuation and trailing spaces from keywords
            words = [s.strip().translate(string.maketrans("", ""), string.punctuation) for s in keyword_string.split(',')]
            # Split keyword phrases into individual words
            for word in words:
                w = word.split(' ')
                words.remove(word)
                for wx in w:
                    words.append(wx.strip().strip('–'))
            for keyword in keywords:
                value = 0
                if keyword in words:
                    if kw_bow in featurenames:
                        value = 1
                    elif kw_tfidf in featurenames:
                        value = 1
                features["KW_" + keyword] = value

        # NARRATIVE features
        if narr_features:
            narr_string = ""
            item = child.find("narrative")
            if item is not None:
                narr_string = item.text.encode("utf-8")
            narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("", ""), string.punctuation).split(' ')]
            #features["narr_length"] = len(narr_words)
            ngrams = {}
            for word in narrwords:
                for x in range(min_ngram, max_ngram):
                    # Cut off the first word
                    grams = ngrams[x]
                    if len(grams > x-1):
                        grams.remove(0)

                value = 0
                if word in narr_words:
                    if 'narr_bow' in featurenames:
                        value = 1
                    else:
                        value = narr_words.count(word)
                fkey = word
                if (not usekeys) or (fkey in keys):
                    features[word] = value
            # Make sure that features align with the training set
            if usekeys:
                for wordkey in keys:
                    if wordkey not in features.keys():
                        features[wordkey] = 0

        # Save features
        matrix.append(features)

    # Convert counts to TFIDF
    if (narr_tfidf in featurenames) or (kw_tfidf in featurenames):
        print('converting to tfidf...')
        count_matrix = []
        matrix_keys = []
        if usekeys:
            for ky in keys:
                if 'CL_' not in ky:
                    matrix_keys.append(ky)
        else:
            for w in narrwords:
                matrix_keys.append(w)
        print('matrix_keys:', str(len(matrix_keys)))

        matrix_keys = sorted(matrix_keys)

        for feat in matrix:
            w_features = []
            for key in matrix_keys:
                w_features.append(feat[key])
            count_matrix.append(w_features)

        tfidfTransformer = sklearn.feature_extraction.text.TfidfTransformer()

        # Use the training count matrix for fitting
        if 'train' in args.infile:
            outf = open("temp.countmatrix", "w")
            outf.write(str(count_matrix))
            outf.close()
            tfidfTransformer.fit(count_matrix)
        else:
            if not os.path.exists("./temp.countmatrix"):
                print('ERROR: train features must be computed first for idf matrix')
                exit(1)
            train_count_matrix = []
            with open("temp.countmatrix", "r") as inf:
                train_count_matrix = eval(inf.read())
            tfidfTransformer.fit(train_count_matrix)

        # Convert matrix to tfidf
        tfidf_matrix = tfidfTransformer.transform(count_matrix)
        print('count_matrix:', str(len(count_matrix)))
        print('tfidf_matrix:', str(tfidf_matrix.shape))

        # Replace features in matrix with tfidf
        for x in range(len(matrix)):
            feat = matrix[x]
            #values = tfidf_matrix[x,0:]
            #print "values: " + str(values.shape[0])
            for i in range(len(matrix_keys)):
                key = matrix_keys[i]
                val = tfidf_matrix[x, i]
                feat[key] = val

    # Write the features to file
    print('writing', str(len(matrix)), 'feature vectors to file...')
    output = open(args.outfile, 'w')
    for feat in matrix:
        #print "ICD_cat: " + feat["ICD_cat"]
        output.write(str(feat) + "\n")
    output.close()

    # Save feature keys
    if kw_features:
        for kw in keywords:
            dict_keys.append("KW_" + kw)
    if narr_features:
        #dict_keys.append("narr_length")
        for w in narrwords:
            dict_keys.append(w)

    kw_output = open(args.outfile + ".keys", "w")
    kw_output.write(str(dict_keys))
    kw_output.close()

def get_keywords(elem):
    keyword_string = ""
    keywords1 = elem.find('CODINGKEYWORDS1')
    if keywords1 is not None:
        keyword_string = keyword_string + keywords1.text.encode("utf-8")
    keywords2 = elem.find('CODINGKEYWORDS2')
    if keywords2 is not None:
        if keyword_string != "":
            keyword_string = keyword_string + ","
        keyword_string = keyword_string + keywords2.text.encode("utf-8")
    return keyword_string.lower()

def add_keywords(keywords, keyword_string, translate_table, stopwords):
    for word in keyword_string.split(','):
        word = word.translate(string.maketrans("", ""), string.punctuation)
        w0 = word.strip()
        if 'kw_phrase' in featurenames:
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


if __name__ == "__main__": main()
