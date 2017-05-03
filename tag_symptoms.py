#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

from lxml import etree
import argparse
import difflib

import extract_features

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./tag_symptoms.py --in [file.xml] --out [outfile.xml]"
        exit()

    run(args.infile, args.outfile)

def run(infile, outfile, usecardinal=True):
    diff_threshold = 0.8
    starttag = '<symptom>'.encode('utf-8')
    endtag = '</symptom>'.encode('utf-8')

    # Load cardinal symptoms file
    print "loading cardinal symptoms"
    cardinal_file = "../../data/cardinal_symptoms.txt"
    keyphrases = []
    if usecardinal:
        with open(cardinal_file, 'r') as f:
            for line in f.readlines():
                keyphrases.append(line.strip())
    
    # Get the xml from file
    print "getting xml"
    tree = etree.parse(infile)
    root = tree.getroot()
    
    for child in root:
        possible_phrases = []
        narr = ""
        # Get the keyword phrases from the record
        keywords = extract_features.get_keywords(child).split(',')
        for kw in keywords:
            keyphrases.append(kw.strip())
        print "keyphrases: " + str(keyphrases)

        # Get the narrative text
        node = child.find("narrative")
        if node != None:
            narr = node.text

        ngrams = get_substrings(narr)
        for keyphrase in keyphrases:
            print "search for: " + keyphrase
            matches = difflib.get_close_matches(keyphrase, ngrams, n=3, cutoff=diff_threshold)
            if len(matches) > 0:
                print "   match: " + matches[0]
                startindex = narr.index(matches[0])
                endindex = startindex + len(matches[0])
                match = [matches[0], startindex, endindex]
                possible_phrases.append(match)

        # Sort possible phrases by start index
        sorted_phrases = sorted(possible_phrases, key=lambda tup: tup[1])
        lastindex = 0
        narr_fixed = ""

        # Resolve conflicts between possible phrases - pick the longest ones
        x = 0
        while x < len(sorted_phrases):
            phrase = sorted_phrases[x]
            start = phrase[1]
            end = phrase[2]
            print "checking: " + str(phrase)
            print "lastindex: " + str(lastindex)
            # Check for overlapping phrases
            if start < lastindex:
                if end <= lastindex:
                    del sorted_phrases[x]
                else:
                    newstart = sorted_phrases[x-1][1]
                    newphrase = [narr[newstart:end], newstart, end]
                    del sorted_phrases[x]
                    sorted_phrases[x-1] = newphrase
            else:
                x = x+1
            lastindex = end
        print "symptoms: " + str(sorted_phrases)

        # Add symptom tags
        lastindex = 0
        for phrase in sorted_phrases:
            phr = phrase[0]
            start = phrase[1]
            end = phrase[2]
            if start > lastindex:
                narr_fixed = narr_fixed + narr[lastindex:start].encode('utf-8')
            narr_fixed = narr_fixed + starttag + " " + narr[start:end].encode('utf-8') + " " + endtag
            lastindex = end

        if lastindex < len(narr):
            narr_fixed = narr_fixed + narr[lastindex:].encode('utf-8')
        
        # Save corrected narrative
        if node == None:
            node = etree.Element("narrative")
        node.text = narr_fixed.strip()
        
    # write the xml to file
    print "writing outfile"
    tree.write(outfile)

def get_ngram(word, prevwords):
    ngram = ""
    for pw in prevwords:
        ngram = ngram + pw + " "
    ngram = ngram + word
    return ngram

def get_substrings(input_string):
    words = input_string.split(' ')
    length = len(words)
    subs = [words[i:j+1] for i in xrange(length) for j in xrange(i,length)]
    #print str(subs)
    substrings = []
    for sub in subs:
        substrings.append(' '.join(sub))
    return substrings

if __name__ == "__main__":main()
