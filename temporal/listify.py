#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Create a temporally ordered list from a graph of events and times

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util3

from lxml import etree
#from xml.sax.saxutils import unescape
import argparse
import networkx as nx
import time

# Local imports
import graphify
import temporal_util as tutil

# Global variables
unk = "UNK"
none_label = "NONE"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    args = argparser.parse_args()

    if not (args.infile):
        print("usage: ./listify.py --in [file_timeml.xml] --relset [binary/simple/exact]")
        exit()

    relset = 'exact'
    if args.relset:
        relset = args.relset
    get_lists(args.infile, relset)

def get_lists(filename, relation_set='exact'):
    print("creating graph: ", relation_set)
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    timelines = []
    ids = []
    records = 0
    dropped = 0

    for child in root:
        id_node = child.find("record_id")
        rec_id = id_node.text
        #print("rec_id:", rec_id)
        records += 1
        node = child.find("narr_timeml_simple")
        #print("node is None:", str(node is None))
        try:
            #node = etree.fromstring(etree.tostring(node).decode('utf8'))
            node = etree.fromstring('<narr_timeml_simple>' + data_util3.stringify_children(node).encode('utf8').decode('utf8') + '</narr_timeml_simple>')
        except etree.XMLSyntaxError as e:
            dropped += 1
            position = e.position[1]
            print("XMLSyntaxError at ", e.position, str(e), data_util3.stringify_children(node)[position-5:position+5])
        if node is not None:
            timeline = listify(node, relation_set)
            ids.append(rec_id)
            timelines.append(timeline)
            records += 1
        # TODO: save timeline to the xml file

    print("records:", str(records))
    print("dropped:", str(dropped))
    # TODO: write xml output to file
    tutil.print_time(time.time()-starttime)
    return timelines

''' Create a graph from xml data
'''
def listify(xml_node, relation_set='exact'):
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))

    # Create the graph
    graph, node_to_ids = graphify.create_digraph(xml_node, relation_set, return_elements=True)

    # Create a binned list based on graph topological order
    timeline = []
    for node in nx.algorithms.dag.topological_sort(graph):
        timeline.append(node_to_ids[node])

    for x in range(len(timeline)):
        print("---", str(x), "---")
        for item in timeline[x]:
            print(item.element.text, '| start:', item.start, 'end:', item.end, etree.tostring(item))

    return timeline

def event_to_string(element):
    return element.attrib['eid'] + ": " + element.text

if __name__ == "__main__":main()
