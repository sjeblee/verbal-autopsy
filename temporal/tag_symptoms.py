#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
#sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
sys.path.append('/u/yoona/ypark_branch/verbal-autopsy')

from lxml import etree
import argparse
import difflib
import os
import subprocess

import data_util
import extract_features
import model_seq

import csv
import pandas as pd
import numpy as np

global symp_narr_tag
symp_narr_tag = "narr_symp"
symp_tagger_tag = "symp_tagger"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--tagger', action="store", dest="tagger")

    # Add an argument for tagging symptoms. 
    #argparser.add_argument('--symptoms', action="store", dest="symptomfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./tag_symptoms.py --in [file.xml] --out [outfile.xml] --train [trainfile.xml] --tagger [tagger] --symptoms [symptomfile.csv]"
        print "tagger options: keyword_match, crf, medttkm, tag_symptoms"
        #if tagger == "tag_symptoms":
        #    if not (args.symptomfile):
        #        print "Must provide the file path to [symptomfile.csv]"
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
        elif tagger == seq2seq:
            tree = seq2seq(tree)
        elif tagger == "tag_symptoms":
            tree = tag_symptoms(tree)
        tree.write(outfile)
        data_util.fix_escaped_chars(outfile)

def crf_tagger(infile, outfile):
    trainfile = '/u/sjeblee/research/data/i2b2/all_timeml_fixed.xml'
    tempfile = './temp.xml'
    model_seq.run(trainfile, infile, tempfile, "crf")
    tree = etree.parse(tempfile)
    tree.write(outfile)
    #newtree = filter_narr(tree, "crf")
    #newtree.write(outfile)

def seq2seq(infile, outfile):
    print "TODO"
    trainfile = '/u/sjeblee/research/data/i2b2/all_timeml_fixed.xml'
    model_seq.run(trainfile, infile, outfile, "seq2seq")

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

def tag_symptoms(tree):
   
    starttag = '<SYMPTOM>'
    endtag = '</SYMPTOM>'
    

    # Open SYMP.csv file
    #csvfile_path = symptomfile
    csvfile_path = "/u/yoona/symptom_files/SYMP.csv" # hard-coded. To be updated. 
    mycsv = csv.reader(open(csvfile_path))

    # Uncomment this to create temp file which contain only the narrative of each data
    #dirpath = os.path.dirname(csvfile_path)
    #narr_temp = dirpath + "/temp.txt"

    # Open tsv file. Comment this if you don't want to add symptoms listed in chv file. 
    chv_tsvfile_path = "/u/yoona/symptom_files/CHV_concepts_terms_flatfile_20110204.tsv" # hard-coded. To be updated
    chv = csv.reader(chv_tsvfile_path, delimiter = '\t')

    symptoms = []

    # Loop over the rows in SYMP.csv file and get the list of symptoms
    for row in mycsv:
        symptoms.append(row[1])
        print(row[1])

    # Loop over the rows in CHV file and get the list of symptoms. 
    # Comment this if you don't want to add symptoms listed in chv file. 
    for row in chv:
        symptoms.append(row[1])
        print(row[1])

    max_word_count = count_max_len_symptoms(symptoms)

    root = tree.getroot()

    for child in root:
        node = child.find("narrative")
        narr = ""
        if node != None:
            narr = node.text.encode('utf-8')
            ngrams = get_substrings_with_limit(narr, max_word_count)
	    ngrams = filter(None, ngrams)
            # Find all phrases that contains words in the list of symptoms. 
            possible_phrases = find_possible_phrases(ngrams, symptoms, narr)

            # Remove duplicates in possible phrases
            clean_possible_phrases = remove_duplicates(possible_phrases)

            # Sort possible phrases by start index
            sorted_phrases = sorted(clean_possible_phrases, key=lambda tup: tup[1])
            
            lastindex = 0
            to_be_tagged_phrases = []
            i = 0
            while i < len(sorted_phrases):
                phrase = sorted_phrases[i]
                start = phrase[1]
                end = phrase[2]
                if i == 0:
                    to_be_tagged_phrases.append(phrase)
                    lastindex = end
                else:
                    if end > lastindex:
                        to_be_tagged_phrases.append(phrase)
                        lastindex = end
                i = i + 1

            lastindex = 0
            narr_symp = ""
            narr_fixed = ""

            for phrase in to_be_tagged_phrases:
                phr = phrase[0]
                start = phrase[1]
                end = phrase[2]
                if start > lastindex:
                    narr_fixed = narr_fixed + narr[lastindex:start]
                narr_fixed = narr_fixed + starttag + " " + narr[start:end] + " " + endtag
                narr_symp = narr_symp + narr[start:end] + " "
                lastindex = end

            if lastindex < len(narr):
                narr_fixed = narr_fixed + narr[lastindex:]

            if node == None:
                node = etree.SubElement(child, "narrative")
            #node.text = narr_fixed.decode('utf-8').strip()
            node = etree.SubElement(child, symp_narr_tag)
            node.text = narr_symp.decode('utf-8').strip()

        '''
            temp = open(narr_temp, "w")
            temp.write("<TEXT>")
            print "narr: " + narr_fixed
            temp.write(narr_fixed)
            temp.write("</TEXT>\n")
            temp.close()
        '''
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

''' Find the symptom with the maximum word count. 
    It is to limit the length of the substring generated from narratives. 
    For time efficiency.
'''
def count_max_len_symptoms(symptoms):

    max_word_count = 0
    for symptom in symptoms:
        curr_count = len(symptom.split(' '))
        if curr_count > max_word_count:
            max_word_count = curr_count

    return max_word_count


''' Generate substrings from input string
    The maximum word count of each substring generated from input string is
    no larger than max_word_count. 
'''
def get_substrings_with_limit(input_string, max_word_count):
    words = input_string.split(' ')
    length = len(words)
    subs = [words[i:j+1] for i in xrange(length) for j in xrange(i,min(length,i + max_word_count))]
    substrings = []
    for sub in subs:
        substrings.append(' '.join(sub))
    return substrings


''' Find all the phrases that contain words describing symptoms. 
'''
def find_possible_phrases(ngrams, symptoms, narr):

    possible_phrases = []
    for substring in ngrams:
        # if the substring ends with punctuation.
        if is_end_index_punc(substring):
            temp_substring = substring[0:len(substring) - 1]
            if (temp_substring.lower() in symptoms):
                startindex = narr.find(substring)
                while startindex != -1:
                    endindex = startindex + len(substring)
                    match = [substring, startindex, endindex]
                    possible_phrases.append(match)
                    startindex = narr.find(substring, endindex)
        
        # if the substring does not end with punctuation. 
        else:
            if (substring.lower() in symptoms):
                startindex = narr.find(substring)
                while startindex != -1:
                    endindex = startindex + len(substring)
                    match = [substring, startindex, endindex]
                    possible_phrases.append(match)
                    startindex = narr.find(substring, endindex)


    return possible_phrases


''' Remove duplicates in possible phrases
'''
def remove_duplicates(phrases):

    clean_possible_phrases = []

    for phrase in phrases:
        if phrase not in clean_possible_phrases:
            clean_possible_phrases.append(phrase)

    return clean_possible_phrases

''' Check whether the input string ends with punctuation. 
'''
def is_end_index_punc(input_string):
    punc = ' :;,./?'
    if input_string[len(input_string)-1] not in punc:
        return False
    return True

if __name__ == "__main__":main()
