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
   
    starttag = '<SYMPTOM>'.encode('utf-8')
    endtag = '</SYMPTOM>'.encode('utf-8')
    # Open csv file
    # csvfile_path = "/Users/yoona96/Desktop/nserc/SYMP.csv"
    csvfile_path = symptomfile
    mycsv = csv.reader(open(csvfile_path))

    dirpath = os.path.dirname(csvfile_path)
    narr_temp = dirpath + "temp.txt"

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
            narr = node.text.encode('utf-8').strip()
	
       	    ngrams = get_substrings_with_limit(narr, max_word_count)
            new_narr = narr.encode('utf-8').strip()

            for substring in ngrams:
                if (substring in symptoms):
                    startindex = new_narr.find(substring)
                    endindex = startindex + len(substring)
                    new_narr = new_narr[0:startindex].encode('utf-8') + starttag + substring.encode('utf-8') + endtag + new_narr[endindex:].encode('utf-8')
            print(new_narr)
	    node.text = new_narr.strip()


            '''if len(new_narr) > 0:
                temp = open(narr_temp, "w")
                temp.write("<TEXT>")
                print "narr: " + new_narr
                temp.write(new_narr)
                temp.write("</TEXT>\n")
                temp.close()
            '''

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
    subs = [words[i:j+1] for i in xrange(length) for j in xrange(i,i + max_word_count)]
    substrings = []
    for sub in subs:
        substrings.append(' '.join(sub))
    return substrings


if __name__ == "__main__":main()
