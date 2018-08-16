#!/usr/bin/python3
# Temporal evaluation functions for thyme

import sys
sys.path.append('..')
sys.path.append('/u/sjeblee/research/git/anaforatools')
import anafora
import tools

import argparse
import os
from copy import deepcopy
from lxml import etree
from xml.sax.saxutils import unescape

thyme_path = '/u/sjeblee/research/data/thyme'
thyme_text = thyme_path + '/text/test'
thyme_ref = thyme_path + '/anafora/test'
thyme_xml = thyme_path + '/test_dctrel.xml'
anafora_dir = '/nbb/sjeblee/thyme/output/system_anafora'

tag_name = 'narr_timeml_crf'

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-o', '--out', action="store", dest="outdir")
    argparser.add_argument('-t', '--test', action="store", dest="testdir")
    args = argparser.parse_args()

    if not (args.infile and args.testdir and args.outdir):
        print("usage: ./thyme_eval.py --in [file_timeml.xml] --out [folder]")
        exit()

        # TODO: convert output tags to inline and split into individual files
        tempfile = args.infile + ".inline"
        tools.to_inline(args.infile, tempfile)
        tools.to_dir(tempfile, args.outdir)

        # Convert reference format to anafora (only need to do this once)
        thyme_dir = thyme_path + '/test_ref'
        tools.to_dir(thyme_xml, thyme_dir)
        anafora.timeml._timeml_dir_to_anafora_dir(thyme_dir, thyme_ref)

        # Run evaluation
        anafora.timeml._timeml_dir_to_anafora_dir(args.outdir, anafora_dir)
        anafora.evaluate.main(["--reference", args.testdir, "--predicted", anafora_dir, "--verbose"])


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