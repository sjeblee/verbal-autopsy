#!/usr/bin/python
# -*- coding: utf-8 -*-
# Add the general COD category to the xml file

from lxml import etree
import argparse
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--map', action="store", dest="mapfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.mapfile):
        print "usage: ./addcategory.py --in [file.xml] --out [outfile.xml] --map [map.csv]"
        exit()

    # Get the ICD mapping
    icdmap = {}
    with open(args.mapfile, "r") as f:
        for line in f:
            tokens = line.split(',')
            icd = tokens[0]
            cat = tokens[1]
            icdmap[icd] = cat.strip()

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    
    for child in root:
        icd = "R99"
        node = child.find("Final_code")
        if node != None:
            icd = node.text
        icdcat = etree.Element("ICD_cat")
        icdcat.text = icdmap[icd]
        child.append(icdcat)
        
    # write the xml to file
    tree.write(args.outfile)

if __name__ == "__main__":main()
