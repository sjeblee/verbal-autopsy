#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Generate distributional stats from xml

from lxml import etree
import argparse
import numpy

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print('usage: ./dist_stats.py --in [file.xml] --out [outfile.csv]')
        exit()

    icd_codes = {}
    icd_cats = {}
    narr_length = {}

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()

    for child in root:
        icd = 'R99'
        node = child.find("Final_code")
        if node is not None:
            icd = node.text
        count = 0
        if icd in icd_codes:
            count = icd_codes[icd]
        icd_codes[icd] = count+1
        node = child.find("cat_who")
        icdcat = node.text
        count = 0
        if icdcat in icd_cats:
            count = icd_cats[icdcat]
        icd_cats[icdcat] = count+1
        node = child.find("narrative")
        narr = ''
        if node is not None:
            narr = node.text
        words = len(narr.split(' '))
        if icdcat in narr_length:
            narr_length[icdcat].append(words)
        else:
            narr_length[icdcat] = [words]

    # write the stats to file
    output = open(args.outfile, "w")
    #output.write("icd_code,num_records\n")
    #for key in icd_codes.keys():
    #    output.write(key + "," + str(icd_codes[key]) + "\n")
    output.write("\nicd_cat,num_records,avg_narr_length\n")
    for key in icd_cats.keys():
        words = narr_length[key]
        avg_narr = numpy.mean(words)
        output.write(key + "," + str(icd_cats[key]) + "," + str(avg_narr) + "\n")
    output.close()


if __name__ == "__main__": main()
