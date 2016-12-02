#!/usr/bin/python
# -*- coding: utf-8 -*-
# Get the text of the narratives

from lxml import etree
import argparse
import numpy
import string
import subprocess

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./getnarrs.py --in [file.xml] --out [outfile.txt]"
        exit()

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()

    narratives = []
    
    for child in root:
        node = child.find("narrative")
        narr = ""
        if node != None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            narratives.append(narr)

    out = open(args.outfile, "w")
    for narr in narratives:
        out.write(narr + "\n")
    out.close()

if __name__ == "__main__":main()
