#!/usr/bin/python
# -*- coding: utf-8 -*-
# Add medttk temporal tags and events to the xml

from lxml import etree
import argparse
import numpy
import os
import string
import subprocess

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./medttk_tag.py --in [file.xml] --out [outfile.xml]"
        exit()

    run(args.infile, args.outfile)

def run(infile, outfile):
    text_infile = "/u/sjeblee/temp.txt"
    text_outfile = "/u/sjeblee/temp-medttk.xml"
    element_name = "narr_medttk"

    print "processing file: " + str(infile)

    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()

    for child in root:
        node = child.find("narrative")
        narr = ""
        if node is not None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            temp = open(text_infile, "w")
            temp.write("<TEXT>")
            temp.write(narr)
            temp.write("</TEXT>\n")
            temp.close()
            # Run medttk parser on narr
            if os.path.exists(text_outfile):
                os.remove(text_outfile)
            process = subprocess.Popen(["python", "/u/sjeblee/tools/medttk/Med-TTK/code/tarsqi.py", "simple-xml ", text_infile, text_outfile], stdout=subprocess.PIPE)
            output, err = process.communicate()

            # Process medttk output file
            medttk_root = etree.parse(text_outfile).getroot()
            med_narr = ""
            for med_child in medttk_root: # sentence
                for item in med_child.iterdescendants("EVENT", "TIMEX3"):
                    if item.text is not None:
                        med_narr = med_narr + " " + item.text
                    else:
                        for it in item.iterdescendants():
                            if it.text is not None:
                                med_narr = med_narr + " " + it.text
                #expressions = med_child.findall('.//EVENT')
                #for ex in expressions:
                #    if ex.text is not None:
                #        med_narr = med_narr + " " + ex.text
            newnode = etree.Element(element_name)
            newnode.text = med_narr
            print "med_narr: " + med_narr
            child.append(newnode)

    # write the stats to file
    tree.write(outfile)


if __name__ == "__main__":main()
