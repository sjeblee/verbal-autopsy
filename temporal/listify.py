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
list_name = "event_list"

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    argparser.add_argument('-o', '--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile):
        print("usage: ./listify.py --in [file_timeml.xml] --out [file.xml] --relset [binary/simple/exact]")
        exit()

    relset = 'exact'
    if args.relset:
        relset = args.relset
    get_lists(args.infile, args.outfile, relset)

def get_lists(filename, outfile, relation_set='exact'):
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
            timeline_node = etree.SubElement(child, list_name)
            timeline_to_xml(timeline, timeline_node)

    print("records:", str(records))
    print("dropped:", str(dropped))
    tree.write(outfile)
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

    print("graph nodes:", str(len(graph.nodes())))
    # Create a binned list based on graph topological order
    timeline = []
    for node in nx.algorithms.dag.topological_sort(graph):
        timeline.append(node_to_ids[node])

    for x in range(len(timeline)):
        print("---", str(x), "---")
        for item in timeline[x]:
            print_event(item)# etree.tostring(item.element))

    return timeline

def print_event(item):
    print(str(item))
    #print(item.eid, ':', item.element.text, ' | start:', item.start, 'end:', item.end)

''' TODO: Make this xml?
'''
def timeline_to_string(timeline):
    string = ""
    for x in range(len(timeline)):
        print("---", str(x), "---")
        for item in timeline[x]:
            string += str(item) + "\n"

def timeline_to_xml(timeline, xml_parent):
    for x in range(len(timeline)):
        for item in timeline[x]:
            event_node = item.element
            event_node.attrib["rank"] = str(x)
            event_node.attrib["start_time"] = item.start
            event_node.attrib["end_time"] = item.end
            xml_parent.append(event_node)
            
            
if __name__ == "__main__":main()
