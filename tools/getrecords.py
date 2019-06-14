#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Get records that match a certain field

from lxml import etree
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument("--field", action="store", dest="field")
    argparser.add_argument('--values', action="store", dest="values")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.field and args.values):
        print('usage: ./getrecords.py --in [file.xml] --out [outfile.xml] --field [Final_code/ICD_cat] --values [A02,B03,C04]')
        exit()

    get_records(args.infile, args.outfile, args.field, args.values)

def get_records(inf, outf, f, vals):
    codeset = vals.split(",")

    # Get the xml from file
    tree = etree.parse(inf)
    root = tree.getroot()

    for child in root:
        val = ""
        node = child.find(f)
        if node is not None:
            val = node.text
        if val not in codeset:
            root.remove(child)

    # write the xml to file
    tree.write(outf)

def get_narrwords(inf, outf, f, vals):
    codeset = vals.split(",")

    # Get the xml from file
    tree = etree.parse(inf)
    root = tree.getroot()
    narrwords = []

    for child in root:
        val = ""
        node = child.find(f)
        if node is not None:
            val = node.text
        if val in codeset:
            nwords = []
            narrnode = child.find("narrative")
            narrtext = ''
            if narrnode is not None:
                narrtext = narrnode.text
            words = narrtext.split()
            for word in words:
                if word not in nwords:
                    nwords.append(word)
            narrwords.append(nwords)

    # write the xml to file
    fileout = open(outf, 'w')
    for vec in narrwords:
        fileout.write(str(vec) + "\n")
    fileout.close()


if __name__ == "__main__": main()
