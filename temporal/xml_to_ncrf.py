#!/usr/bin/python
# -*- coding: utf-8 -*-
# Identify and label symptom phrases in narrative

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import model_seq

from lxml import etree
import argparse
import numpy

id_name = "record_id"
#vecfile = "/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
vecfile = "/u/sjeblee/research/vectors/GoogleNews-vectors-negative300.bin"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./xml_to_ncrf.py --in [file.xml] --out [file.xml]"
        exit()

    extract_features(args.infile, args.outfile)

def extract_features(infile, outfile):
    seq_ids, seqs = model_seq.get_seqs(infile, split_sents=True, inline=False)
    # seqs: list of lists of tuples of (word, label)
    outdata = open(outfile, 'w')
    outids = open(outfile + ".ids", 'w')
    for x in range(len(seqs)):
        seq = seqs[x]
        seqid = seq_ids[x]
        for pair in seq:
            outdata.write(pair[0] + " " + map_label(pair[1]) + "\n")
        outdata.write("\n")
        outids.write(seqid + "\n")
    outdata.close()
    outids.close()

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


if __name__ == "__main__":main()    
