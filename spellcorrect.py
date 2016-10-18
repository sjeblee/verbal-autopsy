#!/usr/bin/python
# -*- coding: utf-8 -*-
# Correct spelling in narrative

from lxml import etree
import argparse
import editdistance
import enchant
import re
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./addcategory.py --in [file.xml] --out [outfile.xml]"
        exit()

    d = enchant.DictWithPWL("en_CA", "dictionary.txt")

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    
    for child in root:
        narr = ""
        #keywords = ""
        node = child.find("narrative")
        if node != None:
            narr = node.text

        # Fix spelling
        narr_words = re.findall(r"[\w']+|[.,!?;]", narr)
        print "narr_words: " + str(narr_words)
        narr_fixed = ""
        for word in narr_words:
            if len(word) > 0:
                if d.check(word):
                    narr_fixed = narr_fixed + " " + word
                else:
                    # TODO: don't change if it's just numbers
                    sugg = d.suggest(word)
                    w = word
                    if (len(sugg) > 0) and editdistance.eval(word, sugg[0]) < 4:
                        w = sugg[0]
                        print word + " -> " + sugg[0]
                    narr_fixed = narr_fixed + " " + w

        # Save corrected narrative
        if node == None:
            node = etree.Element("narrative")
        node.text = narr_fixed.strip()
        #child.append(narr_spell)
        
    # write the xml to file
    tree.write(args.outfile)

if __name__ == "__main__":main()
