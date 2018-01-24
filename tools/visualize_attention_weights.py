#!/usr/bin/python
# -*- coding: utf-8 -*-
# Visualize the attention weights over a narrative and extract phrases?

from lxml import etree
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--records', action="store", dest="infile")
    argparser.add_argument('--ids', action="store", dest="idfile")
    argparser.add_argument('--weights', action="store", dest="weightfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./getnarrs.py --in [file.xml] --out [outfile.txt]"
        exit()

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()

    narratives = {}
    
    for child in root:
        idnode = child.find("MG_ID")
        rec_id = idnode.text
        node = child.find("narrative")
        narr = ""
        if node != None:
            narr = node.text.encode('utf-8')
        narratives[rec_id] = narr

    id_text = open(args.idfile, 'r').read()
    ids = eval(id_text)

    weights = eval(open(args.weightfile, 'r').read())
    output = []
    
    for x in range(len(ids)):
        rec_id = ids[x]
        seq_weights = weights[x]
        seq_len = len(seq_weights)
        narr = narratives[rec_id].split(' ')

        # Trim or pad narrative to max_seq_len
        if len(narr) > seq_len:
            narr = narr[(-1*seq_len):]
        elif len(narr) < seq_len:
            pad_num = seq_len - len(narr)
            for k in range(pad_num):
                narr.insert(0, '0')

        # Match up the weights and words
        for y in range(seq_len):
            word = narr[y]
            w = seq_weights[y]
            output.append(rec_id + "," + word + "," + str(w))

    # Write the output to a file
    out = open(args.outfile, "w")
    for item in output:
        out.write(item + "\n")
    out.close()

if __name__ == "__main__":main()
