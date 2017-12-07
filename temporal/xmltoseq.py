#!/usr/bin/python
# -*- coding: utf-8 -*-
# Convert TimeML tags to sequence labels

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util

from lxml import etree
import argparse

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./xmltoseq.py --in [file.xml] --out [outfile]"
        exit()

    run(args.infile, args.outfile)

def run(infile, outfile):
    seqs = {} # id -> seq
    
    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()
    
    for child in root:
        id_node = child.find("MG_ID")
        rec_id = id_node.text
        node = child.find("narr_timeml_simple")
        narr = ""
        if node != None:
            narr = data_util.stringify_children(node).encode('utf-8')
        if len(narr) > 0:
            seq_narr = xml_to_seq(narr)
            seqs[rec_id] = seq_narr
        
    # write the stats to file
    output = open(outfile, 'w')
    output.write(str(seqs))
    output.close()

def xml_to_seq(text):
    seq = [] # List of tuples
    #print "text: " + text
    BE = 'BE'
    IE = 'IE'
    BT = 'BT'
    IT = 'IT'
    O = 'O'
    event_start = "<EVENT"
    time_start = ["<TIMEX3", "<SECTIME"]
    event_end = "</EVENT>"
    time_end = ["</TIMEX3>", "</SECTIME>"]
    ignore_tags = ["<TLINK", "<SLINK", "<ALINK", "<MAKEINSTANCE"]
    in_event = False
    b_event = False
    b_time = False
    in_time = False
    chunks = text.split(" ")
    #chunk = chunks[0]
    x = 0

    while x < len(chunks):
        chunk = chunks[x].strip()
        if len(chunk) > 0:
            #print "chunk: " + chunk
            # Handle EVENTs
            if in_event:
                if chunk == event_end:
                    in_event = False
                else:
                    word = chunk
                    label = IE
                    if event_end in chunk:
                        ind = chunk.index(event_end)
                        word = chunk[0:ind]
                        in_event = False
                    elif ">" in chunk:
                        ind = chunk.index('>')
                        word = chunk[ind:]
                    if b_event:
                        label = BE
                        b_event = False
                    pair = (word, label)
                    seq.append(pair)
            # Handle TIMEX3
            elif in_time:
                if chunk in time_end:
                    in_time = False
                else:
                    word = chunk
                    label = IT
                    for te in time_end:
                        if te in chunk:
                            ind = chunk.index(te)
                            word = chunk[0:ind]
                            in_time = False
                    if in_time and ">" in chunk:
                        ind = chunk.index('>')
                        word = chunk[ind:]
                    if b_time:
                        label = BT
                        b_time = False
                    pair = (word, label)
                    seq.append(pair)
            elif chunk == event_start:
                in_event = True
                b_event = True
                # Process rest of start tag
                while x < len(chunks) and '>' not in chunks[x]:
                    x = x+1
            elif chunk in time_start:
                in_time = True
                b_time = True
                while x < len(chunks) and '>' not in chunks[x]:
                    x = x+1
            elif chunk in ignore_tags:
                # Ignore the whole tag
                while x < len(chunks) and '>' not in chunks[x]:
                    x = x+1
            else:
                pair = (chunk, O)
                seq.append(pair)
        x = x+1
    #print "seq: " + str(seq)
    return seq

'''
   seqs: dict[id] -> [(word, label),...]
'''
def seq_to_xml(seqs, tag="Adult_Anonymous"):
    root = etree.Element("root")
    for key in seqs:
        seq = seqs[key]
        child = etree.SubElement(root, tag)
        child.text = to_xml(seq)
    tree =  etree.ElementTree(root)
    return tree

def to_xml(seq):
    BE = 'BE'
    IE = 'IE'
    BT = 'BT'
    IT = 'IT'
    O = 'O'
    t_labels = [BT, IT]
    e_labels = [BE, IE]
    event_start = "<EVENT"
    time_start = ["<TIMEX3", "<SECTIME"]
    event_end = "</EVENT>"
                                
    text = ""
    tid = 0
    eid = 0
    prevlabel = O
    for word,label in seq:
        if label == O:
            if prevlabel != O:
                text = text + closelabel(prevlabel)
        elif label == BT:
            text = text + closelabel(prevlabel) + ' <TIMEX3 tid="t' + str(tid) + '">'
            tid = tid+1
        elif label == BE:
            text = text + closelabel(prevlabel) + ' <EVENT tid="e' + str(eid) + '">'
            eid = eid+1

        # Add word
        text = text + ' ' + word
        prevlabel = label
    return text.strip()

def closelabel(prevlabel):
    t_labels = ['BT', 'IT']
    e_labels = ['BE', 'IE']
    text = ""
    if prevlabel in t_labels:
        text = text + ' </TIMEX3>'
    elif prevlabel in e_labels:
        text = text + ' </EVENT>'
    return text

if __name__ == "__main__":main()
