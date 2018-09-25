#!/usr/bin/python3
# Temporal data tools

import os
import re
import subprocess

from copy import deepcopy
from itertools import chain
from lxml.etree import tostring
from lxml import etree
from xml.sax.saxutils import unescape

inline_narr_name = 'narr_timeml_crf'

''' Unescape arrows in an xml file
'''
def fix_arrows(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', filename])


def escape(text):
    text = re.sub('<', '&lt;', text)
    text = re.sub('>', '&gt;', text)
    return text


''' Get content of a tree node as a string
    node: etree.Element
'''
def stringify_children(node):
    #parts = ([str(node.text)] + list(chain(*([tostring(c, encoding='utf-8').decode('utf-8')] for c in node.getchildren()))))
    # filter removes possible Nones in texts and tails
    parts = []
    node_text = node.text if node.text is not None else ""
    node_text = node_text.strip()
    parts.append(node_text)
    for c in node.getchildren():
        parts = parts + list(chain(*([tostring(c, encoding='utf-8').decode('utf-8')])))

    for x in range(len(parts)):
        if type(parts[x]) != str and parts[x] is not None:
            parts[x] = str(parts[x])
        #print("parts[x]:", parts[x])
    return ''.join(filter(None, parts))


''' Convert separate xml tags to inline xml tags
'''
def to_inline(infile, outfile):
    tree = etree.parse(infile)
    treeroot = tree.getroot()

    for child in treeroot:
        docid = child.find("record_id").text
        print(docid)
        #dct = ""
        narr_node = child.find("narrative")
        tag_node = child.find("narr_timeml_simple")
        if narr_node is not None:
            narr = narr_node.text
        tags = tag_node #etree.tostring(tag_node, encoding='utf-8').decode('utf-8')
        tagged_text = insert_tags(narr, tags)
        inline_node = etree.SubElement(child, inline_narr_name)
        inline_node.text = tagged_text

    tree.write(outfile)


'''  Insert xml tags into text based on spans
'''
def insert_tags(text, tags):
    lastindex = 0
    new_text = ""
    for tag in tags:
        if tag.tag == 'TLINK':
            new_text = new_text + etree.tostring(tag, encoding='utf8').decode('utf-8')
        else:
            print(etree.tostring(tag, encoding='utf8').decode('utf-8'))
            span = tag.attrib['span'].split(',')
            start = int(span[0])
            end = int(span[1])
            new_text = new_text + text[lastindex:start] + etree.tostring(tag, encoding='utf8').decode('utf-8')
            lastindex = end
    return new_text


''' Convert inline xml to separate xml files
'''
def to_dir(filename, dirname, node_name):
    print('to_dir:', filename, dirname)
    if not os.path.exists(dirname):
        os.mkdir(dirname)

    treeroot = etree.parse(filename).getroot()

    for child in treeroot:
        docid = child.find("record_id").text
        print(docid)
        dct = ""
        timex_node = etree.Element('TIMEX3')
        narr_node = child.find(node_name)
        if narr_node is not None:
            # Use first identified TIMEX3 as the DCT - TODO: is there a better way to do this?
            orig_node = narr_node.find("TIMEX3")
            timex_node = deepcopy(orig_node)
            timex_node.set("type", "DATE")
            timex_node.set("value", timex_node.text.strip())
            timex_node.set("temporalFunction", "false")
            timex_node.set("functionInDocument", "CREATION_TIME")
            tail = ""
            if timex_node.tail is not None:
                tail = timex_node.tail
            timex_node.tail = ""
            print("tail:", tail)
            dct = unescape(etree.tostring(timex_node, encoding='utf-8', with_tail=False).decode('utf-8'))
            print(dct)
            narr_node.remove(orig_node)
            narr_text = unescape(stringify_children(narr_node))
            if narr_text[0:4] == "None":
                narr_text = narr_text[4:]
            narr = tail + narr_text

            print("narr_text:", narr)

        else:
            print("ERROR: narr is None! Look for: ", node_name)
            #print("in child node:", etree.tostring(child))
            narr = ""
        # Create output tree
        root = etree.XML('<?xml version="1.0"?><TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd"></TimeML>')
        tree = etree.ElementTree(root)
        docid_node = etree.SubElement(root, "DOCID")
        docid_node.text = docid
        dct_node = etree.SubElement(root, "DCT")
        #dct_node.text = dct
        dct_node.append(timex_node)
        text_node = etree.SubElement(root, "TEXT")
        text_node.text = narr
        filename = os.path.join(dirname, docid + ".tml")
        tree.write(filename, encoding='utf-8')

        # Fix arrows
        #fix_arrows(filename)

        # Unsplit punctuation
        unsplit_punc(filename)

def unsplit_punc(filename):
    subprocess.call(["sed", "-i", "-e", 's/ ,/,/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/ \././g', filename])
    subprocess.call(["sed", "-i", "-e", "s/ '/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ :/:/g", filename])
    #subprocess.call(["sed", "-i", "-e", "s/ - /-/g", filename])
    #subprocess.call(["sed", "-i", "-e", "s/\([1-9]\)-\([a-zA-Z]\)/\1 - \2/g", filename])
