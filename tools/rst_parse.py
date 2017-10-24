#!/usr/bin/python
# -*- coding: utf-8 -*-
# Add RST discourse parses to the narrative

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
        print "usage: ./rst_parse.py --in [file.xml] --out [outfile.xml]"
        exit()

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    
    for child in root:
        node = child.find("narrative")
        narr = ""
        if node != None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            tempname = "/u/sjeblee/research/va/data/temp.txt"
            outname = "$gCRF_ROOT$/texts/results/temp/temp.txt.tree"
            temp = open(tempname, "w")
            temp.write(narr)
            temp.close()
            # Run RST parser on narr
            process = subprocess.Popen(["python", "~/tools/RST/src/parse.py", "-t", "temp", tempname], stdout=subprocess.PIPE)
            output, err = process.communicate()
            # Get the output
            with open(outname, 'r') as f:
                lines = f.readlines()
            text = " ".join(lines)
            print text + "\n"
            newnode = etree.Element("narr_rstparse")
            newnode.text = text.decode('utf-8')
            child.append(newnode)
        
    # write the new xml to file
    tree.write(args.outfile)

if __name__ == "__main__":main()
