#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Add the general COD category to the xml file

from lxml import etree
import argparse
import re

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    args = argparser.parse_args()

    if not (args.infile):
        print('usage: ./create_icd_map.py --in [file.csv]')
        exit()

    # Create the ICD mapping
    icdmap = {}
    with open(args.infile, "r") as f:
        for line in f:
            #print "adding: " + line
            tokens = line.split(';')
            cat = tokens[0].strip()
            codes = tokens[2].strip()
            codes = codes.strip("'").strip('"')
            for code in filter(None, re.split("[, ]+", codes)):
                code = code.strip()
                if '-' in code:
                    parts = code.split('-')
                    start = parts[0].strip()
                    end = parts[1].strip()
                    print "expanding range: " + start + " to " + end
                    letter = start[0]
                    startindex = int(start[1:])
                    endletter = end[0]
                    endindex = int(end[1:])
                    if not endletter == letter:
                        print "start and end letter mismatch"
                        endindex1 = 99
                        for k in range(startindex, endindex1+1):
                            code1 = letter + str(k).zfill(2)
                            icdmap[code1] = cat
                            print "added " + code1 + "," + cat
                        letter = endletter
                        startindex = 0
                    for k in range(startindex, endindex+1):
                        code2 = letter + str(k).zfill(2)
                        icdmap[code2] = cat
                        print "added " + code2 + "," + cat
                else:
                    icdmap[code] = cat
                    print "added " + code + "," + cat

    out_codex = open('icd_map_ccodex.csv', 'w')
    for code in icdmap:
        out_codex.write(code + ',' + icdmap[code] + '\n')
    out_codex.close()

    out_codex2 = open('icd_map_ccodex2.csv', 'w')
    for code in icdmap:
        out_codex2.write(code + ',' + icdmap[code][:2] + '\n')
    out_codex2.close()

    out_codex4 = open('icd_map_ccodex4.csv', 'w')
    for code in icdmap:
        out_codex4.write(code + ',' + icdmap[code][0] + '\n')
    out_codex4.close()

if __name__ == "__main__":main()
