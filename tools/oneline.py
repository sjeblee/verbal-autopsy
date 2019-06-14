#!/usr/bin/python3

import argparse
import sys

opentag = ''
closetag = ''

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input', action="store", dest="infile")
    argparser.add_argument('--tag', action="store", dest="tag")
    args = argparser.parse_args()

    if len(args) < 2:
        print('usage: ./oneline.py --input [file] --tag [tag] > [outfile]')

    global opentag
    global closetag
    opentag = "<" + args.tag + ">"
    closetag = "</" + args.tag + ">"
    lastline = None

    with open(args.infile) as xmldata:
        for line in xmldata:
            if lastline is not None:
                parse_line(lastline)
            lastline = line
        parse_line(lastline)

def parse_line(line):
    sys.stdout.write(line.strip())
    if line.strip() == closetag:
        sys.stdout.write('\n')


if __name__ == "__main__": main()
