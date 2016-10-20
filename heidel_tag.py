#!/usr/bin/python
# -*- coding: utf-8 -*-
# Add temporal tags to the xml

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
        print "usage: ./heidel_tag.py --in [file.xml] --out [outfile.xml]"
        exit()

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    
    for child in root:
        node = child.find("narrative")
        narr = ""
        if node != None:
            narr = node.text
        if len(narr) > 0:
            temp = open("/home/sjeblee/temp.txt", "w")
            temp.write(narr)
            temp.close()
            # TODO: run Heideltime tagger on narr
            process = subprocess.Popen(["java", "-jar", "/home/sjeblee/Tools/heideltime/heideltime-standalone/de.unihd.dbs.heideltime.standalone.jar", "/home/sjeblee/temp.txt"], stdout=subprocess.PIPE)
            output, err = process.communicate()
            newnode = etree.Element("narr_tagged")
            newnode.text = output.encode('utf-8')
            child.append(newnode)
        
    # write the stats to file
    tree.write(args.outfile)
    

if __name__ == "__main__":main()
