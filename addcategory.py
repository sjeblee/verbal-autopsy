#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Add the general COD category to the xml file

from lxml import etree
import argparse

import data_util

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--map', action="store", dest="mapfile")
    argparser.add_argument('--label', action="store", dest="label")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.mapfile):
        print("usage: ./addcategory.py --in [file.xml] --out [outfile.xml] --map [map.csv] --label [ICD_cat]")
        exit()

    label = "ICD_cat"
    if args.label:
        label = args.label

    # Get the ICD mapping
    icdmap = data_util.get_icd_map(args.mapfile)

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    removed = 0
    removed_unk = 0
    notfound = []

    for child in root:
        id_node = child.find("MG_ID")
        rec_id = id_node.text
        print('ID:', rec_id)
        icd = 'R99'
        # Skip if record already has an ICD cat
        cat_node = child.find(label)
        overwrite = True
        if (cat_node is None) or overwrite:
            if cat_node is not None:
                child.remove(cat_node)
            node = child.find("Final_code")
            icd = None
            if node is not None and node.text is not None:
                icd = node.text.upper()
            if icd is None or icd == "NULL" or icd == "NR" or icd == "":
                print('Removing record!')
                root.remove(child)
                removed = removed + 1
                continue
            else:
                print('ICD:', icd)
            icdcat = etree.Element(label)
            if icd in icdmap:
                icdcat.text = icdmap[icd]
            else:
                if label == "ICD_cat_neo":
                    icdcat.text = "N5"
                else:
                    print('ICD not found:', icd)
                    notfound.append(icd)
                    root.remove(child)
                    removed_unk = removed_unk + 1
                    continue
            child.append(icdcat)

    # write the xml to file
    tree.write(args.outfile)

    print('Removed', str(removed), 'records with NULL or missing ICD codes')
    print('Removed', str(removed_unk), 'records with unrecognized ICD codes')
    print('ICD codes not recognized:', str(notfound))


if __name__ == "__main__": main()
