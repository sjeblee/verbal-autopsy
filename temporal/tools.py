#!/usr/bin/python3
# Temporal data tools

import subprocess
from itertools import chain
from lxml.etree import tostring
from lxml import etree

def fix_arrows(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', filename])


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


''' Do we need this?
'''
def to_inline(infile, outfile):
    tree = etree.parse(infile)
    treeroot = tree.getroot()

    for child in treeroot:
        docid = child.find("record_id").text
        print(docid)
        dct = ""
        narr_node = child.find("narrative")
        tag_node = child.find("narr_timeml_simple")
        if narr_node is not None:
            narr = narr_node.text
        tags = etree.tostring(tag_node, encoding='utf-8').decode('utf-8')
        tagged_text = insert_tags(narr, tags)
    print("TODO")

    tree.write(outfile)

def insert_tags(text, tags):
    return text

''' Convert inline xml to separate xml files
'''
def to_dir(filename, dirname):
    print("writing eval files...")
    if not os.path.exists(eval_dir):
        os.mkdir(eval_dir)

    treeroot = etree.parse(output_file).getroot()

    for child in treeroot:
        docid = child.find("record_id").text
        print(docid)
        dct = ""
        narr_node = child.find("narr_timeml_crf")
        if narr_node is not None:
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
            narr_text = unescape(tools.stringify_children(narr_node))
            if narr_text[0:4] == "None":
                narr_text = narr_text[4:]
            narr = tail + narr_text

            print("narr_text:", narr)

        else:
            print("narr is None!")
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
        filename = eval_dir + "/" + docid + ".tml"
        tree.write(filename, encoding='utf-8')

        # Fix arrows
        tools.fix_arrows(filename)
        # Unsplit punctuation
        tools.unsplit_punc(filename)

def unsplit_punc(filename):
    subprocess.call(["sed", "-i", "-e", 's/ ,/,/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/ \././g', filename])
    subprocess.call(["sed", "-i", "-e", "s/ '/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ :/:/g", filename])
    #subprocess.call(["sed", "-i", "-e", "s/ - /-/g", filename])
    #subprocess.call(["sed", "-i", "-e", "s/\([1-9]\)-\([a-zA-Z]\)/\1 - \2/g", filename])
