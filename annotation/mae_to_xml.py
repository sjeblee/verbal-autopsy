#!/usr/bin/python
# -*- coding: utf-8 -*-
# Set up annotation files for calculating inter-annotator agreement

from lxml import etree
import argparse
import os
import string
import subprocess

class Tag:
    def __init__(self, name, tid, start, end, text, attributes=""):
        self.name = name
        self.tid = tid
        if start == '':
            self.start = None
        else:
            self.start = int(start)
        if end == '':
            self.end = None
        else:
            self.end = int(end)
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
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--dir', action="store", dest="ann_dir")
    args = argparser.parse_args()

    if not (args.outfile and args.ann_dir):
        print "usage: ./mae_to_mae_in.py --dir [path/to/dir] --in [file.xml] --out [file.xml]"
        exit()

    arg_infile = ""
    if args.infile:
        arg_infile = args.infile

    run(arg_infile, args.outfile, args.ann_dir)

def run(arg_infile, arg_outfile, arg_dir):

    ann_narrs = {} # map from record id to annotated narrative
    for filename in os.listdir(arg_dir):
        temp = filename.split(".")[0]
        #print "temp: " + temp
        if '_' in temp:
            parts = temp.split("_")
            #print "parts: " + str(parts)
            record_name = parts[0]
        else:
            record_name = temp
        print "record_id: " + record_name
        with open(arg_dir + '/' + filename, 'r') as f:
            text = f.read()
            xml_text = convert_spans_to_xml(text)
            ann_narrs[record_name] = xml_text

    # Add the annotated narrative to the xml file
    tree = None
    if arg_infile == "":
        root = etree.Element("Record")
        tree = etree.ElementTree(root)
        for an in ann_narrs:
            child = etree.Element("Adult_Anonymous")
            rec_id = an
            narr = ann_narrs[an]
            id_node = etree.Element("MG_ID")
            id_node.text = rec_id
            child.append(id_node)
            narr_node = etree.Element("narr_timeml_simple")
            narr_node.text = narr.decode('utf-8')
            child.append(narr_node)
            root.append(child)
    else:
        tree = etree.parse(arg_infile)
        root = tree.getroot()
        for child in root:
            id_node = child.find("MG_ID")
            rec_id = id_node.text
            narr_node = etree.Element("narr_timeml_simple")
            narr = ann_narrs[rec_id]
            narr_node.text = narr.decode('utf-8')
            child.append(narr_node)

    tree.write(arg_outfile)
    subprocess.call(["sed", "-i", "-e", 's/&lt;/ </g', arg_outfile])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/> /g', arg_outfile])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', arg_outfile])

def convert_spans_to_xml(text):
    cdata_start = "<TEXT><![CDATA["
    cdata_end = "]]></TEXT>"
    tag_start = "<TAGS>"
    tag_end = "</TAGS>"
    lines = text.splitlines()
    lines = lines[2:] # ignore the xml header
    narr_text = ""
    xml_text = ""
    tags = []
    in_narr = False
    in_tags = False
    x = 0
    while x<len(lines):
        line = lines[x]#.strip()
        if len(line) == 0:
            x = x+1
            continue
        # Get original narr text
        if in_narr:
            if cdata_end in line:
                if len(line) > len(cdata_end):
                    stuff = line[:-(len(cdata_end))]
                    narr_text = narr_text + " " + stuff
                #narr_text = narr_text.strip()
                in_narr = False
            else:
                narr_text = narr_text + " " + line
        # Process tags
        elif in_tags:
            if tag_end in line:
                if len(line) > len(tag_end):
                    stuff = line[:-len(tag_end)]
                    tags.append(create_tag(stuff))
                in_tags = False
            else:
                tags.append(create_tag(line))
        # Find the start of the text
        elif cdata_start in line:
            in_narr = True
            if len(line) > len(cdata_start):
                if cdata_end in line: # Text is all on one line
                    s = len(cdata_start)
                    e = len(line) - len(cdata_end)
                    narr_text = line[s:e]
                    in_narr = False
                else:
                    stuff = line[len(cdata_start):]
                    narr_text = narr_text + " " + stuff
        elif tag_start in line:
            in_tags = True
            if len(line) > len(tag_start):
                stuff = line[len(tag_start):]
                tags.append(create_tag(stuff))
        # Go to the next line        
        x = x+1

    # Sort tags by span start
    tags.sort(key=lambda x: x.start)

    # Construct xml narrative
    i = 0
    tlinks = []
    for z in range(len(tags)):
        tag = tags[z]
        if tag.name == "TLINK":
            tlinks.append(tag)
        else:
            if i < tag.start:
                xml_text = xml_text + narr_text[i:tag.start]
            xml_text = xml_text + str(tag)
            i = tag.end
    if i < len(narr_text):
        xml_text = xml_text + narr_text[i:]
    for t in tlinks:
        xml_text = xml_text + " " + str(t)

    return xml_text

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
    
    tag = Tag(name, tid, start, end, text, attributes)
    return tag

if __name__ == "__main__":main()
