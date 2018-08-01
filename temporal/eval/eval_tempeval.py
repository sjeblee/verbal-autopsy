#!/usr/bin/python3
# Temporal evaluation functions

import sys
sys.path.append('..')
sys.path.append('/u/sjeblee/research/git/tempeval3_toolkit')
import tools

import argparse
import os
import re
import subprocess
from copy import deepcopy
from lxml import etree
from lxml.etree import tostring
from itertools import chain
from xml.sax.saxutils import unescape

none_label = "NONE"
unk = "UNK"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-t', '--test', action="store", dest="testdir")
    argparser.add_argument('-o', '--out', action="store", dest="outdir")
    argparser.add_argument('-y', '--type', action="store", dest="evaltype")
    args = argparser.parse_args()

    if not (args.infile and args.testdir and args.outdir):
        print("usage: ./evaluation.py --in [file_timeml.xml] --test [folder] --out [folder] --type [seq/rel/all]")
        exit()

    eval_seq = True
    eval_rel = True
    if args.evaltype == 'seq':
        eval_rel = False
    elif args.evaltype == 'rel':
        eval_seq = False

    # Run evaluations
    if eval_seq:
        #write_eval_files(args.infile, args.outdir)
        run_tempeval(args.testdir, args.outdir)
    if eval_rel:
        print("TODO: relation evaluation")


def run_tempeval(gold_dir, output_dir):
    print("Running TempEval3 Eval script...")
    pout = subprocess.run(["python", "/u/sjeblee/research/git/tempeval3_toolkit/TE3-evaluation.py", gold_dir, output_dir, "0.5"],
        stdout=subprocess.PIPE, cwd="/u/sjeblee/research/git/tempeval3_toolkit")
    print(pout.stdout.decode('utf-8'))


''' Write output timeml files in TempEval3 format for evalution
    output_file: narrative formatted xml file
    eval_dir: the name of the directory to create with the files for evaluation
'''
def write_eval_files(output_file, eval_dir):
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


if __name__ == "__main__": main()
