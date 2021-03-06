#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Set up annotation files for calculating inter-annotator agreement

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')

from lxml import etree
from xml.sax.saxutils import unescape
import argparse
import os
import subprocess

# Parameters
add_dct_links = False

class ElementTag:
    def __init__(self, start, end, element):
        self.start = start
        self.end = end
        self.element = element

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--textdir', action="store", dest="textdir")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--anndir', action="store", dest="anndir")
    args = argparser.parse_args()

    if not (args.outfile and args.anndir and args.textdir):
        print("usage: ./thyme_to_xml.py --textdir [path/to/dir] --anndir [path/to/dir] --out [file.xml]")
        exit()

    run(args.outfile, args.textdir, args.anndir)

def run(arg_outfile, arg_textdir, arg_anndir):

    txt_narrs = {}
    ann_narrs = {} # map from record id to annotated narrative
    for filename in os.listdir(arg_anndir):
        dotindex = filename.index('.')
        record_name = filename[0:dotindex]
        print("record_id: " + record_name)
        rec_text = ""
        rec_xml = etree.parse(arg_anndir + '/' + filename)
        with open(arg_textdir + '/' + record_name, 'r') as f:
            rec_text = f.read()

        xml_text = convert_spans_to_xml(rec_text, rec_xml)
        ann_narrs[record_name] = xml_text
        txt_narrs[record_name] = rec_text

    # Add the annotated narrative to the xml file
    root = etree.Element("root")
    tree = etree.ElementTree(root)
    for rec_id in ann_narrs:
        child = etree.Element("Record")
        narr = ann_narrs[rec_id]
        id_node = etree.Element("record_id")
        id_node.text = rec_id
        child.append(id_node)
        narr_node = etree.Element("narrative")
        narr_node.text = txt_narrs[rec_id]
        child.append(narr_node)
        narrt_node = etree.Element("narr_timeml_simple")
        narrt_node.text = narr#unescape(narr)
        #print(narrt_node.text)
        child.append(narrt_node)
        root.append(child)

    tree.write(arg_outfile)

    sed_command = r"sed -i -e 's/<Record>/\n<Record>/g' " + arg_outfile
    print("sed_command: ", sed_command)
    ps = subprocess.Popen(sed_command, shell=True, stdout=subprocess.PIPE)
    output = ps.communicate()[0]
    #out = open(arg_outfile, 'w')
    #out.write(output)
    #out.close()

    #subprocess.call(["sed", "-i", "-e", 's/&lt;/ </g', arg_outfile])
    #subprocess.call(["sed", "-i", "-e", 's/&gt;/> /g', arg_outfile])
    #subprocess.call(["sed", "-i", "-e", 's/  / /g', arg_outfile])

def convert_spans_to_xml(raw_text, xml_tree):
    root = xml_tree.getroot()
    annotations = root.find("annotations")
    doc_text = ""
    tlink_text = ""
    tlink_id_num = 1
    spanindex = 0
    dct_id = None
    dct_tlinks = []
    elements = [] # Dict of start span to ElementTags
    time_id_nums = []

    # Find the document creation time
    for entity in annotations:
        entity_id_text = entity.find('id').text
        entity_id = entity_id_text[0:entity_id_text.index('@')]
        entity_type = entity.find('type').text
        dct_types = ['DOCTIME', 'SECTIONTIME']
        if entity_type in dct_types:
            entity_type = 'TIMEX3'
            dct_id = 't' + entity_id
            time_id_nums.append(entity_id)
            print("DCT", str(entity_id))
            span = entity.find('span').text.split(',')
            span_start = int(span[0])
            span_end = int(span[1])
            xml_element = etree.Element(entity_type)
            xml_element.set('tid', dct_id)
            xml_element.set('temporalFunction', "false")
            xml_element.set('functionInDocument', "CREATION_TIME")
            elements.append(ElementTag(span_start, span_end, xml_element))

    for entity in annotations:
        entity_id_text = entity.find('id').text
        entity_id = entity_id_text[0:entity_id_text.index('@')]

        # Process EVENT and TIMEX3 tags
        if entity.tag == "entity":
            span = entity.find('span').text.replace(';', ',').split(',')
            span_start = int(span[0])
            span_end = int(span[1])
            entity_type = entity.find('type').text
            id_type = 'tid'
            if entity_type in dct_types:
                continue
                #entity_type = "TIMEX3"
                #dct_id = entity_id
                #print("DCT", str(entity_id))
            if entity_type == "TIMEX3":
                id_type = 'tid'
                time_id_nums.append(entity_id)
                entity_id = 't' + entity_id
            elif entity_type == "EVENT":
                id_type = 'eid'
                entity_id = 'e' + entity_id
            else:
                print("WARNING: unknown entity type!", entity_type)

            # Get entity attributes
            properties = entity.find('properties')
            xml_element = etree.Element(entity_type)
            xml_element.set(id_type, entity_id)
            if dct_id == entity_id:
                xml_element.set('temporalFunction', "false")
                xml_element.set('functionInDocument', "CREATION_TIME")
            for prop in properties:
                name = prop.tag.lower()
                text = prop.text
                if name == "doctimerel":
                    # Create a TLINK
                    tlink_element = etree.Element("TLINK")
                    #tlink_element.set('lid', 'l' + str(tlink_id_num))
                    #tlink_id_num += 1
                    tlink_element.set('relatedToTime', dct_id)
                    tlink_element.set('relType', text)
                    if entity_type == 'TIMEX':
                        tlink_element.set('timeID', entity_id)
                    else:
                        tlink_element.set('eventID', entity_id)
                    dct_tlinks.append(tlink_element)
                    #tlink_text = tlink_text + str(etree.tostring(tlink_element))
                if text == "N/A":
                    text = "NONE"
                xml_element.set(name, text)

            # Save the element
            if span_start in elements:
                print('WARNING: two tags with the same start span!')
            elements.append(ElementTag(span_start, span_end, xml_element))

        # Process TLINKs
        elif entity.tag == "relation":
            details = False
            tlink_element = etree.Element('TLINK')
            tlink_element.set('lid', 'l' + str(entity_id))
            tlink_id_num = int(entity_id)
            for prop in entity.find('properties'):
                if prop.tag == 'Source':
                    at_index = prop.text.index('@')
                    source_id = prop.text[0:at_index]
                    source_type = prop.text[at_index+1:at_index+2]
                    if source_id in time_id_nums:
                        att_name = 'timeID'
                        tlink_element.set('timeID', 't' + source_id)
                        if details:
                            print('found time')
                    elif source_type == 'e':
                        if details:
                            print('source_type is event')
                        att_name = 'eventID'
                        tlink_element.set('eventID', source_type + source_id)
                    else:
                        print("WARNING: unknown source type!", source_type)
                elif prop.tag == 'Target':
                    at_index = prop.text.index('@')
                    target_id = prop.text[0:at_index]
                    target_type = prop.text[at_index+1:at_index+2]
                    if target_id in time_id_nums:
                        att_name = 'relatedToTime'
                        target_type = 't'
                        if details:
                            print('found time')
                    elif target_type == 'e':
                        if details:
                            print('target_type: event')
                        att_name = 'relatedToEventID'
                    else:
                        print('WARNING: unknown target type!', target_type)
                    tlink_element.set(att_name, target_type + target_id)
                    if details:
                        print('setting', att_name, 'to', target_type + target_id)
                elif prop.tag == 'Type':
                    reltype = prop.text
                    tlink_element.set('relType', reltype)
            if details:
                print('adding tlink:', etree.tostring(tlink_element).decode('utf-8'))
            tlink_text = tlink_text + etree.tostring(tlink_element).decode('utf-8')

    # Add DCT TLINKs
    if add_dct_links:
        for element in dct_tlinks:
            tlink_id_num += 1
            # Check that we haven't already used this id for a DCT relation id
            #if int(entity_id) < tlink_id_num:
            #    print("WARNING: TLINK id collision!")
            #print("adding TLINK", str(tlink_id_num))
            element.set('lid', 'l' + str(tlink_id_num))
            tlink_text = tlink_text + etree.tostring(element).decode('utf-8')

    # Add the tags in order to the text
    #spanindex = 0
    for elementtag in sorted(elements, key=lambda x: x.start):
        span_start = elementtag.start
        span_end = elementtag.end
        element = elementtag.element
        element.set('span', str(span_start) + ',' + str(span_end))
        # Get the text from the spans
        #doc_text = doc_text + raw_text[spanindex:span_start]
        entity_text = raw_text[span_start:span_end]
        element.text = entity_text
        #spanindex = span_end
        doc_text = doc_text + etree.tostring(element).decode('utf-8')

    return doc_text + tlink_text


def remove_duplicate_ann_files(folder):
    rel_suffix = '.Temporal-Relation.gold.completed.xml'
    entity_suffix = '.Temporal-Entity.gold.completed.xml'
    umls_suffix = '.UMLS-Entity.gold.completed.xml'
    for filename in os.listdir(folder):
        dotindex = filename.index('.')
        record_name = filename[0:dotindex]
        print(record_name)
        if os.path.exists(os.path.join(folder, record_name + rel_suffix)): # If the relation file exists
            filepath = os.path.join(folder, record_name + entity_suffix)
            if os.path.exists(filepath):
                os.remove(filepath)
                print('removed entity file')
            filepath = os.path.join(folder, record_name + umls_suffix)
            if os.path.exists(filepath):
                os.remove(filepath)
                print('removed umls file')


if __name__ == "__main__": main()
