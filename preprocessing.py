#!/usr/bin/python
# -*- coding: utf-8 -*-
# Preprocessing functions

from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
import argparse
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--stem', action="store_true", dest="stem")
    argparser.add_argument('--lemma', action="store_true", dest="lemma")
    args = argparser.parse_args()

    if not args.infile and args.outfile:
        print "usage: proprocessing.py --in [file.txt] --out [file.txt] (--stem or --lemma)"
        exit(1)

    text = []
    print "reading file..."
    with open(args.infile, 'r') as f:
        for line in f.readlines():
            text.append(unicode(line, errors='ignore'))
    newtext = []

    if args.stem:
        print "stem"
        for t in text:
            newtext.append(stem(t))
    elif args.lemma:
        for t in text:
            newtext.append(lemmatize(t))

    outfile = open(args.outfile, 'w')
    for line in newtext:
        print line
        outfile.write(line + "\n")
    outfile.close()

def stem(text):
    narr_words = [w.strip() for w in text.lower().split(' ')]
    stemmer = PorterStemmer()
    narr_string = ""
    for nw in narr_words:
        #print "stem( " + nw + ", " + str(len(nw)) + ")"
        newword = stemmer.stem(nw)                                                                                          #print "stem: " + nw + " -> " + newword
        narr_string = narr_string + " " + newword
    return narr_string.strip()

def lemmatize(text):
    lemmatizer = WordNetLemmatizer()
    narr_words = [w.strip() for w in text.lower().split(' ')]
    narr_string = ""
    for nw in narr_words:
        newword = lemmatizer.lemmatize(nw)
        narr_string = narr_string + " " + newword
    return narr_string.strip()

if __name__ == "__main__":main()
