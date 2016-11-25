#!/usr/bin/python
# -*- coding: utf-8 -*-
# Get records that match a certain field

from lxml import etree
import argparse
import getrecords
import string
import subprocess

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--features', action="store", dest="featfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.featfile):
        print "usage: ./getrecords.py --in [file.xml] --out [outfile.csv] --features [feat.txt]"
        exit()

    icdcats = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17"]
    cats = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,15,17]
    field = "ICD_cat"

    with open(args.featfile, 'r') as f:
        feats = f.readlines()

    files = {}
    sensitivity = {}
    specificity = {}
    recs = {}
    rectotals = [0]

    for cat in icdcats:
        out = args.infile + ".narrwords" + cat
        files[cat] = out
        getrecords.get_narrwords(args.infile, out, field, cat)

    for feat in feats:
        index = feats.index(feat)
        feat = feat.strip()
        if feat[0:2] == "W_":
            feat = feat[2:]
        feats[index] = feat
        recs[feat] = [0] # ignore slot 0 so we can use 1-17
        sensitivity[feat] = []
        specificity[feat] = []
        for cat in icdcats:
            recfile = files[cat]
            num = 0
            # Calculate number of records in each category that have this word
            process = subprocess.Popen(["grep", "-c", "\'" + feat + "\'", recfile], stdout=subprocess.PIPE)
            output, err = process.communicate()
            if len(output) > 0:
                num = int(output)
            recs[feat].append(num)

    total = 0
    for cat in cats:
        totalrecs = 0
        for f in recs.keys():
            totalrecs = totalrecs + (recs[f])[cat]
        rectotals.append(totalrecs)
        total = total + totalrecs

    # Calculate sensitivity and specificity
    for feat in feats:
        print "Analyzing feat: " + feat
        featsum = 0
        for cat in cats:
            featsum = featsum + (recs[feat])[cat]
        for cat in cats:
            tp = (recs[feat])[cat]
            p = rectotals[cat]
            print str(cat) + " tp: " + str(tp) + " p: " + str(p)
            sens = 0
            if p > 0:
                sens = tp / p
            spec = (total - featsum - (p - tp)) / total
            sensitivity[feat].append(sens)
            specificity[feat].append(spec)

    # Print the matrices
    fileout = open(args.outfile, 'w')
    fileout.write("0")
    for cat in icdcats:
        fileout.write("," + cat)
    fileout.write("\n")
    for feat in feats:
        fileout.write("sens(" + feat + ")")
        for x in sensitivity[feat]:
            fileout.write("," + str(x))
        fileout.write("\nspec(" + feat + ")")
        for y in specificity[feat]:
            fileout.write("," + str(y))
        fileout.write("\n")
    fileout.close()

if __name__ == "__main__":main()
