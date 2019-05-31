#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Check ICD categories for conflicts

from lxml import etree
import argparse

import data_util

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile1")
    argparser.add_argument('--in2', action="store", dest="infile2")
    argparser.add_argument('--data', action="store", dest="datafile")
    args = argparser.parse_args()

    if not (args.infile1 and args.infile2):
        print('usage: ./crossreference_cats.py --in [file.csv] --in2 [file.csv] --datafile')
        exit()

    # Create the ICD mapping
    icdmap1 = data_util.get_icd_map(args.infile1)
    icdmap2 = data_util.get_icd_map(args.infile2)

    if args.datafile:
        tree = etree.parse(args.datafile)
        root = tree.getroot()
        print('ICD_code,Category,WB10_codex')
        for child in root:
            code = ""
            cat1 = ""
            cat2 = ""
            node = child.find("Final_code")
            if node is not None:
                code = node.text
            else:
                print('Final_code node not found!')
            node = child.find("ICD_cat")
            if node is not None:
                cat1 = node.text
            else:
                print('ICD_cat node not found!')
            node = child.find("Codex_WBD10_adult")
            if node is not None:
                cat2 = node.text
            elif code in icdmap2:
                cat2 = icdmap2[code]
            else:
                print('WB10 node not found!')
                cat2 = "not_found"
            if not cat1 == cat2:
                #print str(code)
                #print str(cat1)
                #print str(cat2)
                if code is not None:
                    print(code, ',cat_', cat1, ',cat_', cat2)

    # See if there are any conflicts
    else:
        print('Conflicts:')
        for entry in icdmap1:
            cat1 = icdmap1[entry]
            cat2 = 'NA'
            if entry in icdmap2:
                cat2 = icdmap2[entry]
            if not cat1 == cat2:
                print(entry, ',cat_', cat1, ',cat_', cat2)


if __name__ == "__main__": main()
