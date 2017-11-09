#!/usr/bin/python
# -*- coding: utf-8 -*-
# Convert full TimeML into simple form

from lxml import etree
import argparse
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./timeml_simplify.py --in [file.xml] --out [outfile.xml]"
        exit()

    run(args.infile, args.outfile)

def run(infile, outfile):
    output = open(outfile, 'w')
    with open(infile, 'r') as f:
        for line in f.readlines():
            output.write(simplify(line) + "\n")
    output.close()

def simplify(text):
    print "text: " + text
    event_start = "<EVENT"
    time_start = "<TIMEX3"
    tlink_start = "<TLINK"
    event_end = "</EVENT>"
    time_end = "</TIMEX3>"
    makeinstance = "<MAKEINSTANCE"
    ignore_tags = ["<SLINK", "<ALINK"]
    lookup = {}
    in_event = False
    in_time = False
    simple_text = ""
    chunks = text.split(" ")
    chunk = chunks[0]
    x = 0

    while x < len(chunks):
        chunk = chunks[x].strip()
        if len(chunk) > 0:
            print "chunk: " + chunk
            # Handle EVENTs
            if in_event:
                if chunk == event_end:
                    in_event = False
                    simple_text = simple_text + " " + chunk
                elif "\"" not in chunk or "eid" in chunk:
                    simple_text = simple_text + " " + chunk
                elif ">" in chunk:
                    simple_text = simple_text + ">"
            # Handle TIMEX3
            elif in_time:
                if chunk == time_end:
                    in_time = False
                    simple_text = simple_text + " " + chunk
                elif "\"" not in chunk or "tid" in chunk:
                    simple_text = simple_text + " " + chunk
                elif ">" in chunk:
                    ind = chunk.index('>')
                    simple_text = simple_text + chunk[ind:]
                    if time_end in chunk:
                        in_time = False
            # Handle MAKEINSTANCE
            elif chunk == makeinstance:
                eid = ""
                eiid = ""
                while "/>" not in chunk and x+1 < len(chunks):
                    x = x+1
                    chunk = chunks[x]
                    if "eventID" in chunk:
                        eid = get_val(chunk)
                    elif "eiid" in chunk:
                        eiid = get_val(chunk)
                lookup[eiid] = eid
            elif chunk in ignore_tags:
                while "</" not in chunk and "/>" not in chunk and x+1 < len(chunks):
                    x = x+1
                    chunk = chunks[x]
            else:
                if chunk == time_start:
                    in_time = True
                    simple_text = simple_text + " " + chunk
                elif chunk == event_start:
                    in_event = True
                    simple_text = simple_text + " " + chunk
                # Handle TLINKS
                elif chunk == tlink_start:
                    queue = chunk
                    add_tlink = False
                    while not "/>" in chunk and x+1 < len(chunks):
                        x = x+1
                        chunk = chunks[x]
                        if "lid" in chunk or "signalID" in chunk:
                            queue = queue + " " + chunk
                        elif "eventInstanceID" in chunk:
                            eiid = get_val(chunk)
                            eventid = lookup[eiid]
                            queue = queue + " eventID=\"" + eventid + "\""
                        elif "relatedToTime" in chunk:
                            queue = queue + " " + chunk
                            add_tlink = True
                        elif "/>" in chunk:
                            queue = queue + "/>"
                    if add_tlink:
                        simple_text = simple_text + " " + queue
                else:
                    simple_text = simple_text + " " + chunk
        x = x+1
    print "simple_text: " + simple_text
    return simple_text.strip()

def get_val(chunk):
    start = chunk.index('"')+1
    end = len(chunk)-1
    return chunk[start:end]
                                                    
if __name__ == "__main__":main()
