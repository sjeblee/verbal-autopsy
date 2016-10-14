#!/usr/bin/python
# -*- coding: utf-8 -*-
# Generate distributional stats from results

import argparse
import numpy
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./results_stats.py --in [file.results] --out [outfile.csv]"
        exit()

    icd_cats = {}

    # Get the xml from file
    with open(args.infile, 'r') as f:
        for line in f:
            res = eval(line)
            cat = res['Predicted_ICD']
            if icd_cats.has_key(cat):
                icd_cats[cat] = icd_cats[cat] + 1
            else:
                icd_cats[cat] = 1
        
    # write the stats to file
    output = open(args.outfile, "w")
    output.write("icd_cat,num_records\n")
    for key in icd_cats.keys():
        output.write(key + "," + str(icd_cats[key]) + "\n")
    output.close()

if __name__ == "__main__":main()
