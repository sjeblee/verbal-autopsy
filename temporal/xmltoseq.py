#!/usr/bin/python
# -*- coding: utf-8 -*-
# Convert TimeML tags to sequence labels

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util

from lxml import etree
import argparse
import re

id_name = "record_id"
symp_narr_tag = "narr_timeml_gru"
#symp_narr_tag = "narr_symp"
BE = 'BE'
IE = 'IE'
BT = 'BT'
IT = 'IT'
O = 'O'

class Element:
    element = None
    start = 0
    end = 0
    def __init__(self, element):
        self.element = element
        span_text = element.attrib['span']
        span = span_text.split(',')
        self.start = int(span[0])
        self.end = int(span[1])

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

'''
Get sequence tags from separate xml tags and text
narr: the text-only narrative
ann: just the xml tags 
split_sents: NOT IMPLEMENTED YET
'''
def ann_to_seq(narr, ann, split_sents):
    ann_element = etree.fromstring("<root>" + ann + "</root>")
    narr_ref = narr.replace('\n', ' ') # Replace newlines with spaces so words get separated
    tags = []
    seqs = [] # a list of tuples (word, label)
    for child in ann_element:
        if child.tag in ['EVENT', 'TIMEX3']:
            tags.append(Element(child))
            #print "element: " + etree.tostring(child).decode('utf8')
    #print "tags: " + str(len(tags))
    # Sort tags by span start
    tags.sort(key=lambda x:x.start)
    index = 0
    for tag in tags:
        if tag.start > index:
            text = narr_ref[index:tag.start]
            index = tag.start
            get_seqs(text, O, seqs)
        if tag.element.tag == 'EVENT':
            label = BE
        elif tag.element.tag == 'TIMEX3':
            label = BT
        for word in split_words(tag.element.text):
            seqs.append((word, label))
            if label == BE:
                label = IE
            elif label == BT:
                label = IT
        index = tag.end
    # Add the tail of the narrative
    if index < len(narr):
        text = narr_ref[index:]
        get_seqs(text, O, seqs)

    # Split sentences
    if split_sents:
        narr = re.sub("\.  ", ". \n", narr) # Add line breaks after sentence breaks
        narr_splits = narr.splitlines()
        #print "split_sents: " + str(len(narr_splits))
        split_seqs = []
        sent_seqs = []
        index = 0
        for chunk in narr_splits:
            chunk = chunk.strip()
            if len(chunk) > 0:
                #print "chunk: " + chunk
                num_words = len(split_words(chunk))
                #print "num_words: " + str(num_words)
                index2 = index+num_words
                sent_seqs = seqs[index:index2]
                #print "seq: " + str(sent_seqs)
                split_seqs.append(sent_seqs)
                index += num_words
        seqs = split_seqs

    return seqs

def get_seqs(text, label, seqs):
    for word in split_words(text):
        seqs.append((word, O))

def split_words(text):
    return re.findall(r"[\w']+|[.,!?;=/\-\[\]]", text.strip())

'''
Convert inline xml-tagged text to sequences
'''
def xml_to_seq(text):
    seq = [] # List of tuples
    #print "text: " + text
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
   filename: optional, the xml file to add the sequences to. If blank, will create a new tree
   tag: the tag to use for new elements if creating a new tree
'''
def seq_to_xml(seqs, filename="", tag="Adult_Anonymous"):
    print "seq_to_xml"
    tree = None
    usefile = False
    if len(filename) >0:
        tree = etree.parse(filename)
        root = tree.getroot()
        usefile = True
        for child in root:
            rec_id = child.find(id_name).text
            seq = ""
            if rec_id in seqs:
                seq = seqs[rec_id]
            narr_node = etree.SubElement(child, symp_narr_tag)
            narr_node.text = to_xml(seq)
    else:
        root = etree.Element("root")
        for key in seqs:
            seq = seqs[key]
            child = etree.SubElement(root, tag)
            narr_node = etree.SubElement(child, symp_narr_tag)
            narr_node.text = to_xml(seq)
            #print "added seq: " + child.text
        tree =  etree.ElementTree(root)
    return tree

def to_xml(seq):
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
        elif label == BT or (label == IT and (prevlabel != BT and prevlabel != IT)):
            text = text + closelabel(prevlabel) + ' <TIMEX3 tid="t' + str(tid) + '">'
            tid = tid+1
        elif label == BE or (label == IE and (prevlabel != BE and prevlabel != IE)):
            text = text + closelabel(prevlabel) + ' <EVENT tid="e' + str(eid) + '">'
            eid = eid+1

        # Add word
        text = text + ' ' + word
        prevlabel = label
    text = text + closelabel(prevlabel)
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
