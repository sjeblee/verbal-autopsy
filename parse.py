#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Add parses to the narrative

from lxml import etree
import argparse
import subprocess

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print('usage: ./parse.py --in [file.xml] --out [outfile.xml]')
        exit()

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()

    for child in root:
        node = child.find("narrative")
        narr = ""
        if node is not None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            tempname = "../mds+rct/narrative/temp.txt"
            temp = open(tempname, "w")
            temp.write(narr)
            temp.close()
            # Run Stanford parser on narr
            process = subprocess.Popen(["java", "-cp", "../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1.jar:../stanford-corenlp-full-2018-02-27/stanford-corenlp-3.9.1-models.jar", "edu.stanford.nlp.parser.nndep.DependencyParser", "-model", "../edu/stanford/nlp/models/parser/nndep/english_SD.gz", "-textFile", tempname, "-outFile", "-"], stdout=subprocess.PIPE)
            output, err = process.communicate()
            newnode = etree.Element("narr_depparse")
            newnode.text = output.decode('utf-8')
            child.append(newnode)

    # write the new xml to file
    tree.write(args.outfile)


if __name__ == "__main__": main()
