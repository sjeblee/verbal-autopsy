#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')

from lxml import etree
import argparse
import difflib
import os
import subprocess

import data_util
import extract_features
import model_crf

global symp_narr_tag
symp_narr_tag = "narr_symp"
symp_tagger_tag = "symp_tagger"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--tagger', action="store", dest="tagger")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./tag_symptoms.py --in [file.xml] --out [outfile.xml] --tagger [tagger]"
        print "tagger options: keyword_match, crf, medttk"
        exit()

    run(args.infile, args.outfile, args.tagger)

def run(infile, outfile, tagger="keyword_match"):
    if tagger == "crf":
        crf_tagger(infile, outfile)
    else:
        tree = etree.parse(infile)
        if tagger == "keyword_match": # DO NOT USE KEYWORD MATCH FOR TEST/DEV
            tree = keyword_match(tree)
        elif tagger == "medttk":
            tree = medttk(tree)
        tree.write(outfile)
        data_util.fix_escaped_chars(outfile)

def crf_tagger(infile, outfile):
    trainfile = '/u/sjeblee/research/data/i2b2/all_timeml_fixed.xml'
    tempfile = './temp.xml'
    model_crf.run(trainfile, infile, tempfile)
    tree = etree.parse(tempfile)
    tree.write(outfile)
    #newtree = filter_narr(tree, "crf")
    #newtree.write(outfile)

def keyword_match(tree, usecardinal=True):
    diff_threshold = 0.8
    starttag = '<EVENT>'.encode('utf-8')
    endtag = '</EVENT>'.encode('utf-8')

    # Load cardinal symptoms file
    cardinal_file = "/u/sjeblee/research/va/data/cardinal_symptoms.txt"
    cardinal = []
    if usecardinal:
        with open(cardinal_file, 'r') as f:
            for line in f.readlines():
                cardinal.append(line.strip())
    
    # Get the xml from file
    root = tree.getroot()

    # TODO: THIS IS CHEATING FOR TEST RECORDS!!!
    for child in root:
        keyphrases = []
        possible_phrases = []
        narr = ""
        # Get the keyword phrases from the record
        keywords = extract_features.get_keywords(child).split(',')
        for kw in keywords:
            keyphrases.append(kw.strip())
        keyphrases = set(keyphrases + cardinal)
        print "keyphrases: " + str(keyphrases)

        # Get the narrative text
        node = child.find("narrative")
        if node != None:
            narr = node.text

        ngrams = get_substrings(narr)
        for keyphrase in keyphrases:
            if len(keyphrase) > 0:
                #print "search for: " + keyphrase
                matches = set(difflib.get_close_matches(keyphrase, ngrams, n=3, cutoff=diff_threshold))
                if len(matches) > 0:
                    for m in matches:
                        #print "   match: " + m
                        startindex = narr.find(m)
                        while startindex != -1:
                            endindex = startindex + len(m)
                            match = [m, startindex, endindex]
                            possible_phrases.append(match)
                            startindex = narr.find(m, endindex)

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
            #print "checking: " + str(phrase)
            #print "lastindex: " + str(lastindex)
            # Check for overlapping phrases
            if start < lastindex:
                if end <= lastindex:
                    if ispunc(narr, end, lastindex):
                        del sorted_phrases[x-1]
                    else:
                        del sorted_phrases[x]
                else:
                    if ispunc(narr, lastindex, end):
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
        narr_symp = ""
        for phrase in sorted_phrases:
            phr = phrase[0]
            start = phrase[1]
            end = phrase[2]
            if start > lastindex:
                narr_fixed = narr_fixed + narr[lastindex:start].encode('utf-8')
            narr_fixed = narr_fixed + starttag + " " + narr[start:end].encode('utf-8') + " " + endtag
            narr_symp = narr_symp + narr[start:end].encode('utf-8') + " "
            lastindex = end

        if lastindex < len(narr):
            narr_fixed = narr_fixed + narr[lastindex:].encode('utf-8')
        
        # Save corrected narrative
        if node == None:
            node = etree.SubElement(child, "narrative")
        node.text = narr_fixed.strip()
        node = etree.SubElement(child, narr_symp_tag)
        node.text = narr_symp.strip()
        tagger_node = etree.SubElement(child, symp_tagger_tag)
        tagger_node.text = "keyword_match"
        
    return tree

def medttk(tree):
    text_infile = "/u/sjeblee/temp.txt"
    text_outfile = "/u/sjeblee/temp-medttk.xml"

    root = tree.getroot()
    for child in root:
        node = child.find("narrative")
        narr = ""
        if node != None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            temp = open(text_infile, "w")
            temp.write("<TEXT>")
            print "narr: " + narr
            temp.write(narr)
            temp.write("</TEXT>\n")
            temp.close()

            # Run medttk parser on narr
            if os.path.exists(text_outfile):
                os.remove(text_outfile)
            process = subprocess.Popen(["python", "/u/sjeblee/tools/medttk/Med-TTK/code/tarsqi.py", "simple-xml ", text_infile, text_outfile], stdout=subprocess.PIPE)
            output, err = process.communicate()

            # Process medttk output file
            medttk_root = etree.parse(text_outfile).getroot()
            med_narr = ""
            for med_child in medttk_root: # sentence
                for item in med_child.iterdescendants("EVENT", "TIMEX3", "TLINK"):
                    # TODO: convert medttk output to simple_timeml
                    if item.text is not None:
                        med_narr = med_narr + " " + start_tag_with_atts(item) + " " + item.text + "</" + item.tag + ">"
                    else:
                        med_narr = med_narr + " " + start_tag_with_atts(item)
                        for it in item.iterdescendants():
                            if it.text is not None:
                                med_narr = med_narr + " " + it.text
                        med_narr = med_narr + "</" + item.tag + ">"
            #med_narr = data_util.stringify_children(medttk_root).decode('utf-8')
            newnode = etree.SubElement(child, symp_narr_tag)
            newnode.text = med_narr.strip()
            tagger_node = etree.SubElement(child, symp_tagger_tag)
            tagger_node.text = "medttk"
            print "med_narr: " + med_narr
    return tree

def start_tag_with_atts(item):
    text = "<" + item.tag
    for key in item.keys():
        text = text + " " + key + '="' + item.get(key) + '"'
    text = text + ">"
    return text

''' Keep only text from inside EVENT and TIMEX3 tags
    tree: the xml tree
    tagger_name: the name of the tool used to identify the events and temporal expressions
'''
def filter_narr(tree, tagger_name):
    print "filter_narr"
    root = tree.getroot()
    symp_narr = ""
    for child in root:
        #print "child: " + data_util.stringify_children(child)
        node = child.find(symp_narr_tag)
        if node == None:
            print "no " + symp_narr_tag + ": " + data_util.stringify_children(child)
        for item in node.iterdescendants("EVENT", "TIMEX3"):
            if item.text is not None:
                symp_narr = symp_narr + " " + item.text
            else:
                for it in item.iterdescendants():
                    if it.text is not None:
                        symp_narr = symp_narr + " " + it.text.strip()
        newnode = etree.SubElement(child, symp_narr_tag)
        newnode.text = symp_narr.strip()
        #print "symp_narr: " + symp_narr
        tagger_node = etree.SubElement(child, symp_tagger_tag)
        tagger_node.text = tagger_name
    return tree

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

def ispunc(input_string, start, end):
    punc = ' :;,./?'
    s = input_string[start:end]
    for char in s:
        if char not in punc:
            return False
    return True

if __name__ == "__main__":main()
