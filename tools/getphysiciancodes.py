#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Get ICD codes from each physician and map them to categories, generate confusion matrix

from lxml import etree
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--map', action="store", dest="mapfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.mapfile):
        print('usage: ./getphysiciancodes.py --in [file.xml] --out [outfile.csv] --map [mapfile]')
        exit()

    get_records(args.infile, args.outfile, args.mapfile)

def get_records(inf, outf, mapf):

    # Get the ICD mapping
    icdmap = {}
    with open(mapf, "r") as f:
        for line in f:
            tokens = line.split(',')
            icd = tokens[0]
            cat = tokens[1]
            icdmap[icd] = cat.strip()

    codes = []

    # Get the xml from file
    tree = etree.parse(inf)
    root = tree.getroot()

    for child in root:
        rec = []
        code1 = ""
        code2 = ""
        codefinal = ""
        node = child.find("CODINGICDCODE1")
        if node is not None:
            code1 = node.text
        node = child.find("CODINGICDCODE2")
        if node is not None:
            code2 = node.text
        node = child.find("Final_code")
        if node is not None:
            codefinal = node.text

        if codefinal == "" or code1 == "" or code2 == "":
            print('Skipping record because of missing code! code1:', code1, 'code2:', code2, 'final:', codefinal)
            node = child.find('MG_ID')
            if node is not None:
                print(node.text)
        else:
            rec.append(icdmap[codefinal])
            rec.append(icdmap[code1])
            rec.append(icdmap[code2])
            codes.append(rec)

    print('processed', str(len(codes)), 'records')

    # Generate confusion matrix
    matrix = []
    for x in range(0, 18):
        row = []
        for y in range(0, 18):
            row.append(0)
        matrix.append(row)

    for item in codes:
        finalcode = int(item[0])
        code1 = int(item[1])
        code2 = int(item[2])

        # If the codes match, mark it as agreement
        if code1 == code2 and code1 == finalcode:
            matrix[finalcode][finalcode] = matrix[finalcode][finalcode] + 1
        elif code1 == code2 and code1 != finalcode:
            print("codes don't match final code: ", str(code1), " ", str(code2), " -> ", str(finalcode))
            matrix[finalcode][code1] = matrix[finalcode][code1] + 1
        elif code1 != finalcode:
            matrix[finalcode][code1] = matrix[finalcode][code1] + 1
        elif code2 != finalcode:
            matrix[finalcode][code2] = matrix[finalcode][code2] + 1

    # write the csv to file
    fileout = open(outf, 'w')
    fileout.write('confusion matrix')
    # Title row
    for x in range(1, 18):
        fileout.write("," + str(x))
    fileout.write("\n")
    # Matrix
    for y in range(1, 18):
        fileout.write(str(y))
        for z in range(1, 18):
            fileout.write("," + str(matrix[y][z]))
        fileout.write("\n")

    fileout.write("\nFinal_code, code1, code2\n")
    for vec in codes:
        for item in vec:
            fileout.write(item + ",")
        fileout.write("\n")
    fileout.close()


if __name__ == "__main__": main()
