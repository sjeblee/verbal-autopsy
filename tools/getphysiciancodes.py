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
    skipped = 0

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

        #print('code1:', code1, 'code2:', code2, 'final_code:', codefinal)

        if codefinal == "" or code1 == "" or code2 == "" or codefinal is None or code1 is None or code2 is None:
            print('Skipping record because of missing code! code1:', code1, 'code2:', code2, 'final:', codefinal)
            node = child.find('MG_ID')
            if node is not None:
                print(node.text)
            skipped += 1
        else:
            cat = 6 # Assign ill-defined to codes not in the map
            if codefinal in icdmap:
                cat_final = icdmap[codefinal]
            else:
                cat_final = cat
            if code1 in icdmap:
                cat_code1 = icdmap[code1]
            else:
                cat_code1 = cat
            if code2 in icdmap:
                cat_code2 = icdmap[code2]
            else:
                cat_code2 = cat
            rec.append(cat_final)
            rec.append(cat_code1)
            rec.append(cat_code2)
            codes.append(rec)

    print('processed', str(len(codes)), 'records, skipped', skipped, 'records')

    # Generate confusion matrix
    maxcat = 7 # num categories +1
    matrix = []
    for x in range(0, maxcat):
        row = []
        for y in range(0, maxcat):
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

    totals = []
    for x in range(0, maxcat):
        cat_total = sum(matrix[x])
        totals.append(cat_total)
    print('totals:', totals)

    # Convert to pecrcentages
    percent_matrix = []
    for x in range(0, maxcat):
        row = []
        total = totals[x]
        for y in range(0, maxcat):
            if x == 0 or y == 0:
                row.append(0)
            else:
                val = matrix[x][y]
                if total == 0:
                    percent = 0
                else:
                    percent = float(val) / total
                row.append(percent)
        percent_matrix.append(row)                            

    # write the csv to file
    fileout = open(outf, 'w')
    fileout.write('confusion matrix')
    # Title row
    for x in range(1, maxcat):
        fileout.write("," + str(x))
    fileout.write("\n")
    # Matrix
    for y in range(1, maxcat):
        fileout.write(str(y))
        for z in range(1, maxcat):
            fileout.write("," + str(matrix[y][z]))
        fileout.write("\n")

    # Percent Matrix
    fileout.write('percent confusion matrix')
    # Title row
    for x in range(1, maxcat):
        fileout.write("," + str(x))
    fileout.write(",total\n")
    for y in range(1, maxcat):
        fileout.write(str(y))
        for z in range(1, maxcat):
            fileout.write("," + str(percent_matrix[y][z]))
        fileout.write("," + str(totals[y]))
        fileout.write("\n")

    fileout.write("\nFinal_code, code1, code2\n")
    for vec in codes:
        for item in vec:
            fileout.write(str(item) + ",")
        fileout.write("\n")
    fileout.close()


if __name__ == "__main__": main()
