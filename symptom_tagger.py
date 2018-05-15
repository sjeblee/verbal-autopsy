#!/usr/bin/python
# -*- coding: utf-8 -*-
# Add parses to the narrative and surround the symptom words from "SYMP.csv" with <symptom>
# and <\symptom>

from lxml import etree
import argparse
import numpy
import string
import subprocess
import csv
import os

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--symptoms', action="store", dest="symptomfile")
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.symptomfile and args.infile and args.outfile):
        print "usage: ./symptoms_parse.py --symptoms [symptomfile.csv] --in [infile.xml] --out [outfile.xml]"
        exit()

    run(args.symptomfile, args.infile, args.outfile)


def run(symptomfile, infile, outfile):
    
    tree = etree.parse(infile)

    tree = narr_tag_symptoms(symptomfile, tree)

    tree.write(outfile)

def narr_tag_symptoms(symptomfile, tree):
   
    starttag = '<SYMPTOM>'
    endtag = '</SYMPTOM>'
    # Open csv file
    # csvfile_path = "/Users/yoona96/Desktop/nserc/SYMP.csv"
    csvfile_path = symptomfile
    mycsv = csv.reader(open(csvfile_path))

    dirpath = os.path.dirname(csvfile_path)
    narr_temp = dirpath + "/temp.txt"

    symptoms = []

    # Loop through the rows in csv file and get the list of symptoms
    for row in mycsv:
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
            node.text = narr_fixed.decode('utf-8').strip()
            node = etree.SubElement(child, "narr_symp")
            node.text = narr_symp.decode('utf-8').strip()

    
            temp = open(narr_temp, "w")
            temp.write("<TEXT>")
            print "narr: " + narr_fixed
            temp.write(narr_fixed)
            temp.write("</TEXT>\n")
            temp.close()

    return tree



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
            if (temp_substring in symptoms):
                startindex = narr.find(substring)
                while startindex != -1:
                    endindex = startindex + len(substring)
                    match = [substring, startindex, endindex]
                    possible_phrases.append(match)
                    startindex = narr.find(substring, endindex)
        
        # if the substring does not end with punctuation. 
        else:
            if (substring in symptoms):
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
