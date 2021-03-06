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

    run(args.infile, args.outfile)

def run(infile, outfile):
    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()

    for child in root:
        node = child.find("narrative")
        narr = ""
        if node is not None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            temp = open("/u/sjeblee/temp.txt", "w")
            temp.write(narr)
            temp.close()
            # Run Heideltime tagger on narr
            process = subprocess.Popen(["java", "-jar", "/u/sjeblee/tools/heideltime/heideltime-standalone/de.unihd.dbs.heideltime.standalone.jar", "/u/sjeblee/temp.txt"], stdout=subprocess.PIPE)
            output, err = process.communicate()
            newnode = etree.Element("narr_heidel")

            # Strip xml header and fix tags
            text0 = output.decode('utf-8')
            lines = text0.split('\n')
            lines = lines[2:]
            text1 = ""
            for line in lines:
                text1 = text1 + line
            newnode.text = text1
            child.append(newnode)

    # write the stats to file
    tree.write(outfile)
    subprocess.call(["sed", "-i", "-e", 's/&lt;/ </g', outfile])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/> /g', outfile])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', outfile])
    subprocess.call(["sed", "-i", "-e", "s/‘/'/g", outfile])
    subprocess.call(["sed", "-i", "-e", "s/’/'/g", outfile])
    subprocess.call(["sed", "-i", "-e", "s/&#8216;/'/g", outfile])
    subprocess.call(["sed", "-i", "-e", "s/&#8217;/'/g", outfile])
    subprocess.call(["sed", "-i", "-e", "s/&#8211;/,/g", outfile])


if __name__ == "__main__": main()
