#!/usr/bin/python
# -*- coding: utf-8 -*-
# Extract features from the xml data
# Feature names: keyword_bow, keyword_tfidf, narr_bow, narr_tfidf

from lxml import etree
import sklearn.feature_extraction
import argparse
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--features', action="store", dest="featurenames")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./raw2vector.py --input [file.xml] --output [outfile] --labels [labelname] --features [f1,f2,f3]"
        print "labels: Final_code, ICD_cat"
        print "features: checklist, kw_bow, kw_tfidf, kw_phrase, narr_bow, narr_tfidf"
        exit()

    labelname = "Final_code"
    if args.labelname:
        labelname = args.labelname

    global featurenames
    rec_type = "type"
    checklist = "checklist"
    kw_phrase = "kw_phrase"
    kw_bow = "kw_bow"
    kw_tfidf ="kw_tfidf"
    narr_bow = "narr_bow"
    narr_count = "narr_count"
    narr_tfidf = "narr_tfidf"
    featurenames = [checklist, kw_bow]
    if args.featurenames:
        featurenames = args.featurenames.split(',')
    print "Features: " + str(featurenames)

    # Set up the keys for the feature vector
    dict_keys = ["MG_ID", labelname]
    if checklist in featurenames:
        dict_keys = dict_keys + ["DeathAge", "ageunit", "DeceasedSex", "Occupation", "Marital", "Hypertension", "Heart", "Stroke", "Diabetes", "TB", "HIV", "Cancer", "Asthma","InjuryHistory", "SmokeD", "AlcoholD", "ApplytobaccoD"]
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

        print "Keywords: " + str(len(keywords))

    # NARRATIVE setup
    if narr_features:
        for child in root:
            narr_string = ""
            node = child.find("narrative")
            if node != None:
                narr_string = node.text.encode("utf-8")
            words = narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')
            for word in words:
                if word not in stopwords:
                    narrwords.add(word.strip())

        print "Words: " + str(len(narrwords))
        
    # Extract features
    matrix = []
    for child in root:
        features = {}

        if rec_type in featurenames:
            features[rec_type] = child.tag

        # CHECKLIST features
        for key in dict_keys:
#            print "- key: " + key
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
            for keyword in keywords:
                value = 0
                if keyword in words:
                    if kw_bow in featurenames:
                        value = 1
                    elif kw_tfidf in featurenames:
                        value = 1 # TODO: calculate tfidf
                features["KW_" + keyword] = value
                
        # NARRATIVE features
        if narr_features:
            narr_string = ""
            item = child.find("narrative")
            if item != None:
                narr_string = item.text.encode("utf-8")
            narr_words = [w.strip() for w in narr_string.lower().translate(string.maketrans("",""), string.punctuation).split(' ')]
            #features["narr_length"] = len(narr_words)
            for word in narrwords:
                value = 0
                if word in narr_words:
                    if "narr_bow" in featurenames:
                        value = 1
                    else:
                        value = narr_words.count(word)
                features["W_" + word] = value

        # Save features
        matrix.append(features)

    # Convert counts to TFIDF
    if (narr_tfidf in featurenames) or (kw_tfidf in featurenames):
        print "converting to tfidf..."
        count_matrix = []
        matrix_keys = []
        for w in narrwords:
            matrix_keys.append("W_" + w)

        for feat in matrix:
            w_features = []
            for key in matrix_keys:
                w_features.append(feat[key])
            count_matrix.append(w_features)

        # Convert matrix to tfidf
        tfidfTransformer = sklearn.feature_extraction.text.TfidfTransformer()
        tfidf_matrix = tfidfTransformer.fit_transform(count_matrix)
            
        # Replace features in matrix with tfidf
        for x in range(len(matrix)):
            feat = matrix[x]
            values = tfidf_matrix[x]
            for i in range(len(matrix_keys)):
                key = matrix_keys[i]
                val = values[i]
                feat[key] = val

    # Write the features to file
    print "writing " + str(len(matrix)) + " feature vectors to file..."
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
            dict_keys.append("W_" + w)

    kw_output = open(args.outfile + ".keys", "w")
    kw_output.write(str(dict_keys))
    kw_output.close() 

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
