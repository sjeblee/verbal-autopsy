#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import model_seq3 as model_seq
import xmltoseq3 as xmltoseq

from lxml import etree
import argparse
import re
import os

id_name = "record_id"
#vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
vecfile = "/u/sjeblee/research/vectors/GoogleNews-vectors-negative300.bin"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--reverse', action="store_true", dest="reverse")
    argparser.set_defaults(reverse=False)
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print("usage: ./xml_to_ncrf.py --in [infile.xml or ncrf_data_dir] --out [file.xml] (--test [testfile.xml] --reverse)")
        exit()

    if args.reverse:
        ncrf_to_xml(args.infile, args.outfile, args.testfile)
    else:
        extract_features(args.infile, args.outfile)


def extract_features(infile, outfile, arg_inline=True):
    seq_ids, seqs = model_seq.get_seqs(infile, split_sents=False, inline=arg_inline, add_spaces=True)
    # seqs: list of lists of tuples of (word, label)
    outdata = open(outfile, 'w')
    outids = open(outfile + ".ids", 'w')
    outorig = open(outfile + ".orig", 'w')
    for x in range(len(seqs)):
        pair = seqs[x]
        #seq = seqs[x]
        seqid = seq_ids[x]
        #for pair in seq:
        print('pair:', pair)
        word = pair[0].strip()
        feats = word_feats(word)
        if word == "$":
            word = "LINEBREAK"
        outorig.write(word + " " + map_label(pair[1]) + "\n")
        # Escape brackets for the NCRF model
        if word == "[":
            word = "LB"
        elif word == "]":
            word = "RB"
        if word not in ['LINEBREAK', 'LB', 'RB']:
            word = word.lower()
        outdata.write(word + " " + feats + ' ' + map_label(pair[1]) + "\n")
        outids.write(seqid + "\n")
        if word == 'LINEBREAK':
            outdata.write('\n')
            outids.write('\n')
            outorig.write('\n')
    outdata.close()
    outids.close()

def word_feats(word):
    num_feat = '[Num]'
    val = 0
    if any(char.isdigit() for char in word):
        val = 1
    return num_feat + str(val)


def map_label(label):
    if label == 'BE':
        return 'B-E'
    elif label == 'IE':
        return 'I-E'
    elif label == 'BT':
        return 'B-T'
    elif label == 'IT':
        return 'I-T'
    else:
        return label


def unmap_label(label):
    return label.replace('-', '')


def ncrf_to_xml(infile, outfile, testfile):
    if os.path.isdir(infile):
        origfile = infile + '/test.bio.orig'
        infile = infile + '/test.out'
    idfile = infile + '.ids'
    ftext = open(infile, 'r')
    fids = open(idfile, 'r')
    forig = open(origfile, 'r')
    textlines = ftext.readlines()

    # Remove lines with hashes
    #print("original textlines:", len(textlines))
    regex = re.compile(r'^# [0-9]')
    textlines = list(filter(lambda i: not regex.search(i), textlines))
    #print("textlines no hash:", len(textlines))
    idlines = fids.readlines()
    origlines = forig.readlines()
    print("textlines:", len(textlines), "orig:", len(origlines), "ids:", len(idlines))
    textlines = list(filter(lambda i: len(i.strip())>0, textlines)) # Remove blank lines
    origlines = list(filter(lambda i: len(i.strip())>0, origlines))
    idlines = list(filter(lambda i: len(i.strip())>0, idlines))
    ftext.close()
    fids.close()
    forig.close()

    id_to_seq = {}

    print("textlines:", len(textlines), "orig:", len(origlines), "ids:", len(idlines))
    #for x in range(0, 10):
    #    print(textlines[x].strip(), "\t|", origlines[x].strip(), "\t|", idlines[x].strip())
    assert(len(textlines) == len(idlines) == len(origlines))
    for x in range(len(textlines)):
        id = idlines[x].strip()
        text = textlines[x].strip()
        orig = origlines[x].strip()

        if (len(text) > 0) and (len(id) > 0): # Ignore blank lines
            if len(text) == 0:
                text = 'LINEBREAK O'
            text = text.split(' ')

            if id not in id_to_seq:
                id_to_seq[id] = []
            word = text[0]
            if word == 'LB':
                word = '['
            elif word == 'RB':
                word = ']'
            # Get word from origlines and make sure it matches the current output line
            orig_word = orig.split(' ')[0]
            print("orig:", orig_word, "word:", word)
            assert(orig_word.lower() == word.lower())
            #if orig_word == 'LB':
            #    orig_word = '['
            #elif orig_word == 'RB':
            #    orig_word = ']'
            tag = unmap_label(text[1])
            id_to_seq[id].append((orig_word, tag))

    tree = xmltoseq.seq_to_xml(id_to_seq, testfile, tag="narr_timeml_ncrf")
    tree.write(outfile)


if __name__ == "__main__": main()
