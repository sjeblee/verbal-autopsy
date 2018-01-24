#!/usr/bin/python
# -*- coding: utf-8 -*-
# Correct spelling in narrative

from lxml import etree
import argparse
import editdistance
import enchant
import kenlm
import re
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--lm', action="store_true")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./spellcorrect.py --in [file.xml] --out [outfile.xml] (--lm)"
        exit()

    if args.lm:
        run(args.infile, args.outfile, args.lm)
    else:
        run(args.infile, args.outfile)

def run(infile, outfile, arg_lm=False):
    d = enchant.DictWithPWL("en_CA", "/u/sjeblee/research/va/git/verbal-autopsy/dictionary.txt")
    mapping = {'labor':'labour', 'laborer':'labourer', 'color':'colour', 'yeras':'years', 'elergies':'allergies', 'around12':'around 12', 'learnt':'learned', 'rigor':'rigour', 'couldn':'couldn\'t', 'didnt':'didn\'t', 'didn':'didn\'t', 'neighbor':'neighbour', 'enjury':'injury'}

    # Language model
    #lmfile = None
    #lm = None
    #if arg_lm:
    lmfile = "/u/sjeblee/research/data/ICE-India/ice-lm-5.binary"
    lm = kenlm.Model(lmfile)

    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()
    
    for child in root:
        narr = ""
        #keywords = ""
        node = child.find("narrative")
        if node != None:
            narr = node.text

        # If the whole narrative is uppercase, lowercase it
        if narr.isupper():
            narr = narr.lower()

        # Fix spelling, unless it's a number or it's capitalized
        narr_words = re.findall(r"[\w']+|[.,!?;]", narr)
        print "narr_words: " + str(narr_words)
        narr_fixed = ""
        prevwords = []
        prevn = 4
        for word in narr_words:
            if len(word) > 0:
                # Hand-crafted mappings
                if word in mapping.keys():
                    narr_fixed = narr_fixed + " " + mapping[word]
                    prevwords.append(mapping[word])
                    print word + " -> " + mapping[word]
                elif d.check(word) or word.isdigit() or word.isupper() or word.istitle() or (len(word) < 3):
                    narr_fixed = narr_fixed + " " + word
                    prevwords.append(word)
                else:
                    # Split XXX from other words
                    if word[0:3] == "XXX" and len(word) > 3:
                        word = "XXX " + word[3:]
                    elif word[-3:] == "XXX" and len(word) > 3:
                        word = word[0:-3] + " XXX"
                    sugg = d.suggest(word)
                    bestw = word
                    ngram = get_ngram(word, prevwords)
                    bestp = lm.score(ngram, bos=False, eos=False)
                    bested = 3
                    print "orig: " + ngram + " : " + str(bestp)

                    if arg_lm:
                        for s in sugg:
                            ed = editdistance.eval(word, s)
                            if (len(s) > 0) and (ed <= bested):
                                # Favor corrections with lower edit distances
                                if ed < bested:
                                    bested = ed
                                ngram = get_ngram(s, prevwords)
                                prob = lm.score(ngram, bos=False, eos=False)
                                print "try: " + ngram + " : " + str(prob)
                                if prob > bestp:
                                    bestp = prob
                                    bestw = s
                    else:
                        if len(sugg) > 0 and editdistance.eval(word, sugg[0]) < bested:
                            bestw = sugg[0]
                    print word + " -> " + bestw
                    narr_fixed = narr_fixed + " " + bestw
                    prevwords.append(bestw)
                if len(prevwords) > prevn:
                    del prevwords[0]

        # Save corrected narrative
        if node == None:
            node = etree.Element("narrative")
        node.text = narr_fixed.strip()
        #child.append(narr_spell)
        
    # write the xml to file
    tree.write(outfile)

def get_ngram(word, prevwords):
    ngram = ""
    for pw in prevwords:
        ngram = ngram + pw + " "
    ngram = ngram + word
    return ngram

if __name__ == "__main__":main()
