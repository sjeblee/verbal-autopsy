#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Create a graph from the temporal relations in the data

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util3

from lxml import etree
import argparse
import pydot
import networkx as nx
import time

# Local imports
import temporal_util as tutil
from temporal_util import Event, TimeRel

# Global variables
debug = True
unk = "UNK"
none_label = "NONE"
node_to_ids = {}
id_to_node = {}
before_types = ['BEFORE']
overlap_types = ['OVERLAP', 'CONTAINS']

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-r', '--relset', action="store", dest="relset")
    args = argparser.parse_args()

    if not (args.infile):
        print("usage: ./graphify.py --in [file_timeml.xml] --relset [binary/simple/exact]")
        exit()

    relset = 'exact'
    if args.relset:
        relset = args.relset
    get_graphs(args.infile, relset)

def get_graphs(filename, relation_set='exact'):
    print("creating graph: ", relation_set)
    starttime = time.time()
    # Get the xml from file
    tree = etree.parse(filename)
    root = tree.getroot()
    graphs = []
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
            graph = create_graph(node, relation_set)
            digraph = create_digraph(node, relation_set)
            ids.append(rec_id)
            graphs.append(digraph)
            records += 1

            # Save an image of the graph
            pdot_graph = nx.drawing.nx_pydot.to_pydot(graph)
            pdot_graph.write_png(filename + ".graph.png")
            pdot_digraph = nx.drawing.nx_pydot.to_pydot(digraph)
            pdot_digraph.write_png(filename + ".digraph.png")

    # Print the first few feature vectors as a sanity check
    print("records:", str(records))
    print("dropped:", str(dropped))
    print("time:", str(time.time()-starttime), "s")
    return graphs

''' Create a graph from xml data
'''
def create_graph(xml_node, relation_set='exact', return_event_map=False):
    events = xml_node.xpath("EVENT")
    times = xml_node.xpath("TIMEX3")
    tlinks = xml_node.xpath("TLINK")
    print("events: ", str(len(events)), " times: ", str(len(times)), " tlinks: ", str(len(tlinks)))

    # Create the graph
    graph = nx.DiGraph()
    event_map = {}
    timex_map = {}
    for event in events:
        if 'eid' not in event.attrib:
            print("no eid: ", etree.tostring(event))
        event_id = event.attrib['eid']
        graph.add_node(event_id, label=event.text)
        #print("adding event: ", event_id, event.text)
        event_map[event_id] = Event(event_id, event)

    for timex in times:
        time_id = timex.attrib['tid']
        #print("adding time: ", time_id, timex.text)
        graph.add_node(time_id, label=timex.text)
        timex_map[time_id] = timex

    print("nodes:", str(len(graph.nodes())))
    print("events:", str(len(event_map.keys())))
    print("times:", str(len(timex_map.keys())))

    # Add event relations to the graph
    for tlink in tlinks:
        # event-event relations
        if 'relatedToEventID' in tlink.attrib and 'eventID' in tlink.attrib:
            event_id = tlink.attrib['eventID']
            event2_id = tlink.attrib['relatedToEventID']
            rel_type = tlink.attrib['relType']
            rel_type = tutil.map_rel_type(rel_type, relation_set)
            if event_id not in event_map or event2_id not in event_map:
                print("WARNING: missing event:", event_id)
                #event_map[event_id] = Event(event_id, None)
            #if event2_id not in event_map:
            #    print("adding event from tlink:", event2_id)
            #    event_map[event2_id] = Event(event2_id, None)
            # Flip after relations when adding to the graph
            id1, id2, rel_type = tutil.flip_relations(event_id, event2_id, rel_type)
            graph.add_edge(id1, id2, label=rel_type)

    # Add time information to events, and add transitive time-event relations
    for tlink in tlinks:
        rel_type = tlink.attrib['relType']
        # event-time
        if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
            event_id = tlink.attrib['eventID']
            time_id = tlink.attrib['relatedToTime']
            # Flip relations if necessary
            id1, id2, rel_type = tutil.flip_relations(event_id, time_id, rel_type)
        # time-event
        elif 'timeID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            event_id = tlink.attrib['relatedToEventID']
            time_id = tlink.attrib['timeID']
            id1, id2, rel_type = tutil.flip_relations(time_id, event_id, rel_type)
        # event-event relations handled in the next section
        else:
            continue
        #elif 'timeID' in tlink.attrib and 'relatedToTimeID' in tlink.attrib:
        #    time_id = tlink.attrib['timeID']
        #    time2_id = tlink.attrib['relatedToTimeID']
        #    id1, id2, rel_type = tutil.flip_relations(time_id, event_id, rel_type)
        graph.add_edge(id1, id2, label=rel_type)
        
        # Add time value to the event
        e = event_map[event_id]
        add_time_to_event(e, timex_map[time_id], rel_type)

    # Add transitive event relations (BEFORE only)
    for tlink in tlinks:
        # Transitive time-event before relations
        if 'relatedToTime' in tlink.attrib and 'eventID' in tlink.attrib:
            event_id = tlink.attrib['eventID']
            time_id = tlink.attrib['relatedToTime']
            rel_type = tlink.attrib['relType']
            if rel_type in before_types:
                for other_event in graph.successors(time_id):
                    second_rel = graph.get_edge_data(time_id, other_event)
                    if not graph.has_edge(event_id, other_event) and second_rel in before_types:
                        graph.add_edge(event_id, other_event, label='BEFORE')

        # Propagate time information across contains relations
        elif 'timeID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            event_id = tlink.attrib['relatedToEventID']
            time_id = tlink.attrib['timeID']
            rel_type = tlink.attrib['relType']
            if rel_type == 'CONTAINS':
                for other_event in graph.successors(event_id):
                    second_rel = graph.get_edge_data(time_id, other_event)
                    if (not (graph.has_edge(time_id, other_event) or graph.has_edge(other_event, time_id))) and second_rel == 'CONTAINS':
                        add_time_to_event(other_event, timex_map[time_id], rel_type)

        # Transitive event-event relations
        elif 'eventID' in tlink.attrib and 'relatedToEventID' in tlink.attrib:
            # Handle transitive ends-on etc. relations
            event_id = tlink.attrib['eventID']
            event2_id = tlink.attrib['relatedToEventID']
            rel_type = tlink.attrib['relType']
            if rel_type == 'BEGINS-ON':
                for nex in graph.successors(event2_id):
                    if graph.get_edge_data(event2_id, nex) == 'OVERLAP' and nex in timex_map:
                        if event_map[event_id].start == 'UNK':
                            event_map[event_id].start = tutil.time_value(timex_map[nex])
                for prev in graph.predecessors(event2_id):
                    if graph.get_edge_data(prev, event2_id) == 'CONTAINS' and prev in timex_map:
                        if event_map[event_id].start == 'UNK':
                            event_map[event_id].start = tutil.time_value(timex_map[prev])
            elif rel_type == 'ENDS-ON':
                for nex in graph.successors(event2_id):
                    if graph.get_edge_data(event2_id, nex) == 'OVERLAP' and nex in timex_map:
                        if event_map[event_id].end == 'UNK':
                            event_map[event_id].end = tutil.time_value(timex_map[nex])
                for prev in graph.predecessors(event2_id):
                    if graph.get_edge_data(prev, event2_id) == 'CONTAINS' and prev in timex_map:
                        if event_map[event_id].end == 'UNK':
                            event_map[event_id].end = tutil.time_value(timex_map[prev])
                
    # Add event relations based on start and end times
    for eid in event_map.keys():
        for eid2 in event_map.keys():
            if eid != eid2 and (not graph.has_edge(eid, eid2)) and (not graph.has_edge(eid2, eid)):
                event1 = event_map[eid]
                event2 = event_map[eid2]
                rel = tutil.compare_events(event1, event2)
                #if debug: print("event-event rel:", rel)
                if rel == 'BEFORE':
                    graph.add_edge(eid, eid2, label='BEFORE')
                    if debug: print("found time BEFORE relation: ", str(event1), "|", str(event2))
                elif rel == 'AFTER':
                    graph.add_edge(eid2, eid, label='BEFORE')
                    if debug: print("found time BEFORE relation: ", str(event2), "|", str(event1))
                elif rel == 'OVERLAP':
                    graph.add_edge(eid, eid2, label='OVERLAP')
                    # Make sure we haven't just created a cycle
                    try:
                        nx.algorithms.cycles.find_cycle(graph, eid)
                        #if debug: print('not adding edge because it would create a cycle')
                        graph.remove_edge(eid, eid2)
                    except:
                        if debug: print("found time OVERLAP relation:", str(event1), "|", str(event2))
                        pass

    # Resolve cycles:
    graph, num_cycles = resolve_cycles(graph)
    print("num_cycles:", str(num_cycles))

    if return_event_map:
        print("event_map events:", str(len(event_map.keys())))
        return graph, event_map, timex_map
    else:
        return graph

def create_digraph(xml_node, relation_set='exact', return_elements=False):
    graph, event_map, timex_map = create_graph(xml_node, relation_set, True)

    # NOT WORKING - Remove duplicate events from the graph?
    #nodelist = list(graph.nodes())
    #for node_id in nodelist:
    #    node = event_map[node_id]
    #    if graph.has_node(node_id):
    #        for node2_id in nodelist:
    #            if graph.has_node(node2_id):
    #                node2 = event_map[node2_id]
    #                if node.equals(node2) and check_edges(node_id, node2_id, graph):
    #                    graph.remove_node(node2_id)
    #print("nodes after de-duplication:", str(len(graph.nodes())))

    # Remove time nodes from the graph
    #print("Removing time nodes...")
    #for tid in timex_map:
    #    graph.remove_node(tid)
    print("Event nodes in original graph:", str(len(graph.nodes())))

    # Create a directed graph based on simple relation set
    # Nodes in this graph will be lists of events
    node_num = 0
    digraph = nx.DiGraph()
    for (node,neigh,data) in graph.edges(data=True):
        node_found = node in id_to_node
        neigh_found = neigh in id_to_node
        node_id = node_num
        if node_found:
            node_id = id_to_node[node]
        #elif neigh_found:
        #    node_id = id_to_node[neigh]
        else:
            node_id = node_num
            node_num += 1
        # Put overlapping events in the same node only if there's no other incoming links
        if data['label'] in overlap_types and graph.in_degree(neigh) <= 1 and graph.out_degree(neigh) <= 1:
            update_node(digraph, node, node_id)
            update_node(digraph, neigh, node_id)
        else: #data['label'] == 'BEFORE':
            if not node_found:
                update_node(digraph, node, node_num)
                node_num += 1
            if not neigh_found:
                update_node(digraph, neigh, node_num)
                node_num += 1
            startnode = id_to_node[node]
            endnode = id_to_node[neigh]
            if startnode != endnode:
                digraph.add_edge(startnode, endnode, label=data['label'])

    # Remove timex ids from the digraph
    for item in id_to_node.keys():
        if item[0] == 't':
            node_to_ids[id_to_node[item]].remove(item)
            id_to_node[item] = None

    # Delete empty nodes
    for node in list(digraph.nodes()):
        if len(node_to_ids[node]) == 0:
            digraph.remove_node(node)

    # TODO: what about nodes with no edges?
    for node in list(graph.nodes()):
        if node not in id_to_node:
            print("node with no edges dropped:", str(event_map[node]))

    # Combine nodes with the same in and out relations
    nid = 0
    node_list = list(digraph.nodes(data=True))
    while nid < len(node_list):
        if node_list[nid] is not None:
            node, data = node_list[nid]
            for nid2 in range(len(node_list)):
                node2 = None
                if node_list[nid2] is not None:
                    node2, data2 = node_list[nid2]
                if (node2 is not None) and (node2 != node):
                    # If the two nodes have thesame relations, combine them
                    same = True # Do node and node2 have the same relations?
                    for nextnode in digraph.successors(node):
                        edge_data = digraph[node][nextnode]
                        if 'label' not in edge_data:
                            print('no label:', str(node), str(nextnode), str(edge_data))
                        rel = edge_data['label']
                        if (not digraph.has_edge(node2, nextnode)) or (digraph[node2][nextnode]['label'] != rel):
                            same = False
                    for prevnode in digraph.predecessors(node):
                        edge_data = digraph[prevnode][node]
                        if 'label' not in edge_data:
                            print('no label:', str(node), str(nextnode), str(edge_data))
                        rel = edge_data['label']
                        if (not digraph.has_edge(prevnode, node2)) or (digraph[prevnode][node2]['label'] != rel):
                            same = False
                    if same:
                        combine_nodes(digraph, node, node2)
                        # Remove the combined node, but keep nid the same
                        node_list[nid2] = None
        # Move to the next node
        nid += 1

    # Check for cycles
    for cycle in nx.simple_cycles(digraph):
        print("cycle found!", str(cycle))
        break

    if return_elements:
        num_elements = 0
        for nid in node_to_ids:
            num_elements += len(node_to_ids[nid])
        print("num_elements:", str(num_elements))
        node_to_event = {}
        for nid in node_to_ids:
            #print("nid:", nid)
            node_to_event[nid] = []
            for eid in node_to_ids[nid]:
                #print("- eid:", eid)
                node_to_event[nid].append(event_map[eid])
        return digraph, node_to_event
    else:
        return digraph

#------------ Helper Functions ------------

''' Add time information to an event object
    event: Event object
    time: xml element representing a TIMEX
    rel_type: the relation type (presumably from event to time)
'''
def add_time_to_event(event, time, rel_type):
    tag = ""
    tval = tutil.time_value(time)
    if rel_type == 'BEGINS-ON':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.ON
            tag = "start"
    elif rel_type in 'ENDS-ON':
        if event.end == unk:
            event.end = tval
            event.end_type = TimeRel.ON
            tag = "end"
    elif rel_type == 'BEFORE':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.BEFORE
            tag = "start"
        if event.end == unk:
            event.end = tval
            event.end_type = TimeRel.BEFORE
            tag += "/end"
    elif rel_type == 'BEFORE/OVERLAP':
        if event.start == unk:
            event.start = tval
            event.start_type = TimeRel.BEFORE
        if event.overlap == unk:
            event.overlap = tval
            tag = "start/overlap"
    elif rel_type in overlap_types:
        if event.start == unk and event.end == unk:
            event.overlap = tval
            tag = "overlap"
    else:
        print("unhandled rel type:", rel_type)
    print("added time to event:", tag, tval, "-->", str(event))

''' Check to see if two nodes have the same relation edges
    graph: the directed graph
'''
def check_edges(node1, node2, graph):
    # Check that they have the same relations
    for pre in graph.predecessors(node1):
        if not graph.has_edge(pre, node2):
            return False
        rel = graph.get_edge_data(pre, node1)
        if not graph.get_edge_data(pre, node2) == rel:
            return False
    for nex in graph.successors(node1):
        if not graph.has_edge(node2, nex):
            return False
        rel = graph.get_edge_data(node1, nex)
        if not graph.get_edge_data(node2, nex) == rel:
            return False
    return True

''' Combine two nodes
'''
def combine_nodes(digraph, node1, node2):
    #print("combine_nodes", str(node1), str(node2))
    for eid in node_to_ids[node2]:
        update_node(digraph, eid, node1)
    digraph.remove_node(node2)

''' Resolve cycles by dropping overlap links
'''
def resolve_cycles(graph):
    count = 0
    for cycle in nx.simple_cycles(graph):
        count += 1
        fixed = False
        print("cycle found!", str(cycle))
        edges = nx.algorithms.cycles.find_cycle(graph, source=cycle)
        for from_node, to_node in edges:
            edge_data = graph.get_edge_data(from_node, to_node)
            if edge_data['label'] == 'OVERLAP':
                graph.remove_edge(from_node, to_node)
                fixed = True
        if fixed:
            print("cycle resolved by removing OVERLAP link")
        else:
            print("WARNING: cycle could not be resolved!")
    return graph, count

def update_node(digraph, eid, node_num):
    #print("update node:", eid, str(node_num))
    # Make sure the node is in the graph
    if node_num not in node_to_ids:
        digraph.add_node(node_num)
        node_to_ids[node_num] = []
    # Collapse two nodes if necessary
    if eid in id_to_node:
        # If the event is already in the right node, do nothing
        if id_to_node[eid] == node_num:
            return
        # Move other members of the old node to the new node
        old_node = id_to_node[eid]
        for eidn in node_to_ids[old_node]:
            node_to_ids[node_num].append(eidn)
            id_to_node[eidn] = node_num
        node_to_ids[old_node] = []
        # Move edges from old node to new node
        for eidn in list(digraph.predecessors(old_node)):
            edge_label = digraph[eidn][old_node]['label']
            digraph.remove_edge(eidn, old_node)
            if eidn != node_num:
                digraph.add_edge(eidn, node_num, label=edge_label)
        for eidn in list(digraph.successors(old_node)):
            edge_label = digraph[old_node][eidn]['label']
            digraph.remove_edge(old_node, eidn)
            if eidn != node_num:
                digraph.add_edge(node_num, eidn, label=edge_label)    
    # Add the eid to the correct node
    id_to_node[eid] = node_num
    if eid not in node_to_ids[node_num]:
        node_to_ids[node_num].append(eid)


if __name__ == "__main__":main()
