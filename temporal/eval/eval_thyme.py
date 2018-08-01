#!/usr/bin/python3
# Temporal evaluation functions for thyme

import sys
sys.path.append('/u/sjeblee/research/git/anaforatools')
import anafora

import argparse
import os
import re
import subprocess
from copy import deepcopy
from lxml import etree
from lxml.etree import tostring
from itertools import chain
from xml.sax.saxutils import unescape

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-o', '--out', action="store", dest="outdir")
    argparser.add_argument('-t', '--test', action="store", dest="testdir")
    args = argparser.parse_args()

    if not (args.infile and args.testdir and args.outdir):
        print("usage: ./thyme_eval.py --in [file_timeml.xml] --out [folder]")
        exit()

        # TODO: convert tags to inline
        tempfile = args.infile + ".inline"
        to_inline(args.infile, tempfile)

        # Write TE3 formatted evaluation files
        evaluation.write_eval_files(tempfile, args.outdir)

        # Run evaluation
        anafora_dir = ""
        _timeml_dir_to_anafora_dir(args.outdir, anafora_dir)
        anafora.evaluate.main(["--reference", anafora_dir])

def to_inline(infile, outfile):
    print("TODO")
