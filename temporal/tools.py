#!/usr/bin/python3
# Temporal data tools

import subprocess
from itertools import chain
from lxml.etree import tostring

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
    print("TODO")

def to_dir(filename, dirname):
    print("TODO")

def unsplit_punc(filename):
    subprocess.call(["sed", "-i", "-e", 's/ ,/,/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/ \././g', filename])
    subprocess.call(["sed", "-i", "-e", "s/ '/'/g", filename])
    #subprocess.call(["sed", "-i", "-e", "s/ - /-/g", filename])
    #subprocess.call(["sed", "-i", "-e", "s/\([1-9]\)-\([a-zA-Z]\)/\1 - \2/g", filename])
