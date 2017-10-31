#!/usr/bin/python
# -*- coding: utf-8 -*-
# Set up annotation files for calculating inter-annotator agreement

from lxml import etree
import argparse
import string

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--dir', action="store", dest="ann_dir")
    args = argparser.parse_args()

    if not (args.infile and args.ann_dir):
        print "usage: ./xml_to_mae_out.py --in [file_ann.xml] --dir [path/to/dir]"
        exit()

    inname = args.infile
    temp = inname.split(".")[0]
    print "temp: " + temp
    parts = temp.split("_")
    print "parts: " + str(parts)
    ann_name = parts[len(parts)-1]
    print "ann_name: " + ann_name

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    for child in root:
        id_node = child.find("MG_ID")
        rec_id = id_node.text
        narr_text_node = child.find("narrative")
        narr_text = narr_text_node.text
        narr_node = child.find("narr_timeml_simple")
        narr = etree.tostring(narr_node)
        # Extract tags and spans
        narr = narr[20:(len(narr)-21)]
        print "narr: " + narr
        x = 0
        index = 0
        narr_tags = ""
        while x < len(narr):
            
            char = narr[x]
            if char == '<':
                startspan = index
                x = x+1
                if len(narr) > x+5:
                    tag = narr[x:x+6]
                    print "found tag: " + tag
                    idnum = ""
                    # Fix longer tags
                    if tag == "TIMEX3" or tag == "SIGNAL":
                        x = x+1
                    tag = tag.strip()
                    # Skip space and id tag name
                    x = x+11
                    # Get id
                    end = narr.index('"', x)
                    idnum = narr[x:end]
                    x = end
                    # Create tag string
                    restoftag = ""
                    char = narr[x]
                    while char != '>' and x < len(narr):
                        x = x+1
                        char = narr[x]
                        if char != ">":
                            restoftag = restoftag + char
                    if tag != "TLINK":
                        # Calcuate span
                        text = ""
                        # Skip initial space
                        x = x+1
                        if narr_text[index] == " ":
                            index = index+1
                        print "x: " + str(x) + " i: " + str(index)
                        print "narr[x]: '" + str(narr[x]) + "' narr[i]: '" + str(narr_text[index]) + "'"
                        while char != '<' and x < len(narr):
                            x = x+1
                            index = index+1
                            char = narr[x]
                            text = text + char
                        endspan = index-2
                        index = index-2 # remove trailing space
                        text = text[:-2]
                        print "text from x: '" + text + "'"
                        text = narr_text[startspan:endspan].strip()
                        print "text from i: '" + text + "'"
                        print "x: " + str(x) + " i: " + str(index)
                        #narr_text = narr_text + text + " "
                        narr_tags = narr_tags + '<' + tag + ' id="' + idnum + '" spans="' + str(startspan) + '~' + str(endspan) + '" text="' + text + '" ' + restoftag + "/>\n"
                        # Skip over end tag
                        while char != '>' and x < len(narr):
                            x = x+1
                            char = narr[x]
                    else: # TLINK TODO: create TLINKs from event attributes
                        from_start = restoftag.index('eventID') +9
                        from_end = restoftag.index('"', from_start)
                        from_id = restoftag[from_start:from_end]
                        to_start = restoftag.index('relatedToTime') +15
                        to_end = restoftag.index('"', to_start)
                        to_id = restoftag[to_start:to_end]
                        print "TLINK from: " + from_id + " to: " + to_id
                        narr_tags = narr_tags + '<' + tag + ' id="' + idnum + '" from="' + from_id + '" to="' + to_id + '" ' + restoftag + ">\n"
                    x = x+1 #Skip the space after the tag
                    #char = narr[x]
                    #if (char == " "):
                    #    x = x+1
                    #    char = narr[x]
                    #    print "x: " + str(x) + " char: " + char
            x = x+1
            index = index+1
                
        filename = args.ann_dir + "/" + rec_id + "_" + ann_name + ".xml"
        print "writing " + filename
        outfile = open(filename, 'w')
        outfile.write('<?xml version="1.0" encoding="UTF-8" ?>\n<simpleTimeML>\n')
        outfile.write('<TEXT><![CDATA[')
        outfile.write(narr_text)
        outfile.write(']]></TEXT>\n<TAGS>')
        outfile.write(narr_tags)
        outfile.write("\n</TAGS>\n</simpleTimeML>\n")
        #outfile.write("</root>\n")


if __name__ == "__main__":main()
