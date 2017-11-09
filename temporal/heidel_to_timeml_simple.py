#!/usr/bin/python
# -*- coding: utf-8 -*-
# Generate useful info from heideltime tags and depparses

from timeml_simplify import simplify

from lxml import etree
from lxml.etree import tostring
from itertools import chain
import argparse
import numpy
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./heidel_to_timeml_simple.py --in [file_heidel.xml] --out [outfile.xml]"
        exit()

    global temp_start, temp_end
    temp_start = "<TIMEX3"
    temp_end = "</TIMEX3>"

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    
    for child in root:
        node = child.find("narrative")
        narr = ""
        narr_ht = ""
        simple_text = ""
        if node != None:
            narr = node.text.encode('utf-8')
            simple_text = narr
        if len(narr) > 0:
            # Get the TimeML node
            node_ht = child.find("narr_heidel")
            if node_ht != None:
                node_timeml = node_ht.find("TimeML")
                narr_ht = stringify_children(node_timeml)#.encode('utf-8')
                print "narr_ht: " + narr_ht
                simple_text = simplify(narr_ht)
                text = simple_text.replace("&lt;", "<").replace("&gt;", ">")
            # Create a new node for timeml_simple
            newnode = etree.Element("narr_timeml_simple")
            newnode.text = simple_text
            child.append(newnode)
        
    # write the new xml to file
    tree.write(args.outfile)

def stringify_children(node):
    parts = ([node.text] + list(chain(*([tostring(c)] for c in node.getchildren()))))
    for part in parts:
        if part is not None:
            print "part: " + part
    # filter removes possible Nones in texts and tails
    return ''.join(filter(None, parts))

if __name__ == "__main__":main()
