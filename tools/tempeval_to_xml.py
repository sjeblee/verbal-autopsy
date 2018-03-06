#!/usr/bin/python
# -*- coding: utf-8 -*-
# Set up annotation files for calculating inter-annotator agreement

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util

from lxml import etree
import argparse
import os
import string
import subprocess

class Tag:
    def __init__(self, name, tid, text, attributes=""):
        self.name = name
        self.tid = tid
        self.text = text
        self.attributes = attributes

    def __str__(self):
        string = ""
        if self.name == "TLINK":
            string = '<' + self.name + ' id="' + self.tid + '" ' + self.attributes + '/>'
        else:
            string = '<' + self.name + ' id="' + self.tid + '" ' + self.attributes + '> ' + self.text + ' </' + self.name + '>'
        return string

def main():
    argparser = argparse.ArgumentParser()
    #argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--dir', action="store", dest="ann_dir")
    args = argparser.parse_args()

    if not (args.outfile and args.ann_dir):
        print "usage: ./tempeval_to_xml.py --dir [path/to/dir] --out [file.xml]"
        exit()

    run(args.outfile, args.ann_dir)

def run(arg_outfile, arg_dir):

    txt_narrs = {}
    ann_narrs = {} # map from record id to annotated narrative
    for filename in os.listdir(arg_dir):
        record_name = filename.split(".")[0]
        #print "temp: " + temp
        #if '_' in temp:
        #    parts = temp.split("_")
        #    #print "parts: " + str(parts)
        #    record_name = parts[0]
        #else:
        #    record_name = temp
        print "record_id: " + record_name
        with open(arg_dir + '/' + filename, 'r') as f:
            text = f.read()
            narr_text, xml_text = convert_spans_to_xml(text)
            ann_narrs[record_name] = xml_text
            txt_narrs[record_name] = narr_text

    # Add the annotated narrative to the xml file
    root = etree.Element("Record")
    tree = etree.ElementTree(root)
    for rec_id in ann_narrs:
        child = etree.Element("Adult_Anonymous")
        narr = ann_narrs[rec_id]
        id_node = etree.Element("MG_ID")
        id_node.text = rec_id
        child.append(id_node)
        narr_node = etree.Element("narrative")
        narr_node.text = txt_narrs[rec_id]
        child.append(narr_node)
        narrt_node = etree.Element("narr_timeml_simple")
        narrt_node.text = narr.decode('utf-8')
        child.append(narrt_node)
        root.append(child)

    tree.write(arg_outfile)
    subprocess.call(["sed", "-i", "-e", 's/&lt;/ </g', arg_outfile])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/> /g', arg_outfile])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', arg_outfile])

def convert_spans_to_xml(text):
    lines = text.splitlines()
    lines = lines[1:] # ignore the xml header
    root  = etree.fromstring("<root>" + ' '.join(lines) + "</root>")
    timeml = root.find("TimeML")
    chunks = []

    # Get DCT
    dct_node = timeml.find("DCT")
    dct_text = data_util.stringify_children(dct_node)

    # Get document text
    text_node = timeml.find("TEXT")
    text_text = dct_text + data_util.stringify_children(text_node)
    print "text: " + text_text

    #build_list = etree.XPath("//")
    chunks = text_to_list(text_text)
    print "chunks: " + str(chunks)
    
    narr_text = ""
    xml_text = ""
    tags = []
    in_narr = False
    in_tags = False

    # Sort tags by span start
    #tags.sort(key=lambda x: x.start)

    # Construct xml narrative
    #i = 0
    #tlinks = []
    #for z in range(len(tags)):
    #    tag = tags[z]
    #    if tag.name == "TLINK":
    #        tlinks.append(tag)
    #    else:
    #        if i < tag.start:
    #            xml_text = xml_text + narr_text[i:tag.start]
    #        xml_text = xml_text + str(tag)
    #        i = tag.end
    #if i < len(narr_text):
    #    xml_text = xml_text + narr_text[i:]
    #for t in tlinks:
    #    xml_text = xml_text + " " + str(t)

    return narr_text, xml_text

def create_tag(line):
    timex = "<TIMEX3"
    event = "<EVENT"
    signal = "<SIGNAL"
    tlink = "<TLINK"
    sectime = "<SECTIME"
    #attributes = ["polarity=", "relatedToTime"]

    name = ""
    tid = ""
    start = ""
    end = ""
    text = ""
    attributes = ""

    print line

    chunks = line.split(' ')

    x = 0
    
    while x<len(chunks):
        chunk = chunks[x]
        if timex in chunk:
            name = "TIMEX3"
        elif event in chunk:
            name = "EVENT"
        elif signal in chunk:
            name = "SIGNAL"
        elif tlink in chunk:
            name = "TLINK"
        elif sectime in chunk:
            name = "SECTIME"
        elif "spans=" in chunk:
            string = chunk[7:-1]
            parts = string.split('~')
            start = int(parts[0])
            end = int(parts[1])
            print "start: " + str(start) + ", end: " + str(end)
        elif "start=" in chunk:
            start = int(chunk[7:-1])
        elif "end=" in chunk:
            end = int(chunk[5:-1])
        elif "text=" in chunk:
            chunk = chunk[6:]
            text = ""
            while '"' not in chunk:
                text = text + chunk + " "
                x = x+1
                chunk = chunks[x]
            text = text + chunk[:-1]
        elif "id=" in chunk:
            tid = chunk[4:-1]
        elif chunk != "/>":
            attributes = attributes + " " + chunk
        x = x+1

    # TODO: process TLINKS?
    print "name: " + name + ", tid: " + tid + ", text: " + text + ", atts: " + attributes
    
    tag = Tag(name, tid, text, attributes)
    return tag

def text_to_list(text):
    chunks = []
    string = ""
    in_tag = False
    in_close_tag = False
    for char in text:
        print "char: " + char
        if char == '<' and not in_tag:
            print "in_tag"
            in_tag = True
            chunks.append(string)
            string = char
        elif char == '<' and in_tag:
            print "in_close_tag <"
            in_close_tag = True
            string = string + char
        elif char == "/" and in_tag and not in_close_tag:
            print "in_close_tag /"
            in_close_tag = True
            string = string + char
        elif char == '>' and in_close_tag:
            print "close tag"
            string = string + char
            chunks.append(string)
            string = ""
            in_tag = False
            in_close_tag = False
        else:
            string = string + char
    if len(string) > 0:
        chunks.append(string)
    return chunks

if __name__ == "__main__":main()
