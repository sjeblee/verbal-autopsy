#!/usr/bin/python3
# Temporal data tools

import os
import re
import shutil
import subprocess

from copy import deepcopy
from itertools import chain
from lxml.etree import tostring
from lxml import etree
from xml.sax.saxutils import unescape

#inline_narr_name = 'narr_ctakes'
debug = True

def adjust_spans(inline_xml, outfile, inline_narr_name, ref_dir=None):
    print('adjust_spans')
    fix_arrows_only(inline_xml)
    inline_tree = etree.parse(inline_xml)

    for child in inline_tree.getroot():
        rec_id = child.find("record_id").text
        print("adjust_spans", rec_id)
        narr = unescape(stringify_children(child.find(inline_narr_name)))
        #narr = stringify_children(unescape(etree.tostring(child.find(inline_narr_name), encoding='utf-8').decode('utf-8')))
        ref_narr = child.find('narrative').text

        # Use a reference text file instead of the narr from the xml
        if ref_dir is not None:
            with open(os.path.join(ref_dir, rec_id), 'r') as f:
                ref_narr = escape(f.read())

        new_narr = fix_narr_spans(narr, ref_narr)

        print('Saving to xmltree...')
        narr_node = child.find(inline_narr_name)
        narr_node.text = new_narr

        # Update spans
        '''
        points = sorted(offset_map.keys())
        print('offset keys:', len(points))
        key_index = 0
        key = points[key_index]
        for tag in narr_node:
            print('updating tag:', tag.tag)
            # Get span
            if "span" not in tag.attrib:
                print("ERROR: no span:", etree.tostring(tag))
            span = tag.attrib["span"].split(',')
            start = span[0]
            end = span[1]

            # Update the span start
            while start > key:
                key_index += 1
                key = points[key_index]
            off = offset_map[key]
            start = start + off

            while end > key:
                key_index += 1
                key = points[key_index]
            off = offset_map[key]
            end = end + off
            print('updating span:', span, str(start) + "," + str(end))
            tag.set('span', str(start) + "," + str(end))
        child.remove(narr_node)
        child.append(narr_node_new)
        '''
    # Write the results back to file
    inline_tree.write(outfile)

def fix_narr_spans(narr, ref_narr):
    offset_map = {}
    offset = 0
    index = 0
    ref_index = 0
    new_narr = ''
    insert_chars = ['\n', '"', '#', ' ', ':', ';', '(', ')', '*', '%', '&', '-', '+', '$', '[', ']', '`', '@']
    count = 0

    # Fix the text and save offsets
    while ref_index < len(ref_narr):
        ref_char = ref_narr[ref_index]
        count = 0

        # Check for the end of the pred narr
        if index >= len(narr):
            new_narr = new_narr + ref_narr[ref_index:]
            return new_narr

        char = narr[index]
        if debug: print('char  [', index, ']', char, 'ref[', ref_index, ']', ref_char, 'offset', offset)

        while not char == ref_char:
            count += 1
            if count > 20:
                print('WARNING: same ref char for 20 iterations:', ref_char)
            # Close tags before inserting missing chars
            if char == '<' and narr[index+1] == '/': # Ignore tags
                if debug: print('ignoring end tag')
                char_buffer = ''
                while not char == '>':
                    char_buffer = char_buffer + char
                    #new_narr = new_narr + char
                    index += 1
                    char = narr[index]
                    if debug: print('ignoring [', index, ']', char)
                # Add the close >
                char_buffer = char_buffer + char
                if debug: print('closed [', index, ']', char)
                index +=1
                if 'TimeML' not in char_buffer:
                    # Delete trailing spaces from inside the tags
                    save_char = ''
                    while new_narr[-1] in insert_chars:
                        save_char = save_char + new_narr[-1]
                        new_narr = new_narr[0:-1]
                        if debug: print('moved trailing char')
                    new_narr = new_narr + char_buffer + save_char
                else:
                    if debug: print('Dropped TimeML tag')

                # Check for the end of the pred narr
                if index >= len(narr) or char_buffer == '</TimeML>':
                    new_narr = new_narr + ref_narr[ref_index:]
                    return new_narr

                char = narr[index]

            elif ref_char in insert_chars:
                new_narr = new_narr + ref_char
                if debug: print('added [', index, ']', char, 'ref[', ref_index, ']', ref_char, 'offset', offset)
                ref_index += 1
                offset = offset+1

                # Fix escaped < and >
                if ref_char == '&':
                    if debug: print('checking for gt and lt')
                    if ref_index < len(ref_narr)-2 and ((ref_narr[ref_index:ref_index+3] == 'gt;' and not (index > len(narr)-2 and narr[index:index+3] == 'gt;')) or (ref_narr[ref_index:ref_index+3] == 'lt;' and not (index > len(narr)-2 and narr[index:index+3] == 'lt;'))):
                        new_narr = new_narr + ref_narr[ref_index:ref_index+3]
                        if debug: print('added [', index, ']', char, 'ref[', ref_index, ']', ref_narr[ref_index:ref_index+1], 'offset', offset)
                        ref_index += 3
                        offset += 3

                offset_map[index] = offset

                # Check for the end of the ref narr
                if ref_index >= len(ref_narr):
                    return new_narr
                ref_char = ref_narr[ref_index]
                count = 0
                #continue

            # Add open tags
            elif char == '<': # Ignore tags
                if debug: print('ignoring start tag')
                char_buffer = ''
                while not char == '>':
                    char_buffer = char_buffer + char
                    index += 1
                    char = narr[index]
                    if debug: print('ignoring [', index, ']', char)
                # Add the close >
                char_buffer = char_buffer + char
                if debug: print('closed [', index, ']', char)
                index +=1
                if 'TimeML' not in char_buffer:
                    new_narr = new_narr + char_buffer
                else:
                    if debug: print('Dropped TimeML tag')

                # Check for the end of the pred narr
                if index >= len(narr):
                    new_narr = new_narr + ref_narr[ref_index:]
                    return new_narr

                char = narr[index]

            else:
                # Move forward until we find the next char
                if debug: print('check [', index, ']', char, 'ref[', ref_index, ']', ref_char, 'offset', offset)
                if char == ref_char:
                    count = 0
                    break
                index += 1
                offset -= 1

                # Check for the end of the pred narr
                if index >= len(narr):
                    new_narr = new_narr + ref_narr[ref_index:]
                    return new_narr

                char = narr[index]
                # Found match
                offset_map[index] = offset

        if debug: print('match [', index, ']', char, 'ref[', ref_index, ']', ref_char, 'offset', offset)
        new_narr = new_narr + char
        ref_index += 1
        index += 1

    return new_narr


''' Unescape arrows in an xml file
'''
def fix_arrows(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', filename])

def fix_arrows_only(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/</g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/>/g', filename])

def escape(text):
    text = re.sub('<', '&lt;', text)
    text = re.sub('>', '&gt;', text)
    return text

def escape_and(text):
    return re.sub('&', '&amp;', text)

def stringify_node(node):
    return unescape(etree.tostring(node, encoding='utf-8').decode('utf-8'))


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
def to_inline(infile, outfile, narr_name='narr_timeml_crf'):
    tree = etree.parse(infile)
    treeroot = tree.getroot()

    for child in treeroot:
        docid = child.find("record_id").text
        print('to_inline:', docid)
        #dct = ""
        narr_node = child.find("narrative")
        tag_node = child.find("narr_timeml_simple")
        if narr_node is not None:
            narr = narr_node.text
        tags = etree.fromstring(unescape(etree.tostring(tag_node, encoding='utf-8').decode('utf-8'))) # fix escaped arrows
        tagged_text = insert_tags(narr, tags)
        inline_node = etree.SubElement(child, narr_name)
        inline_node.text = tagged_text

    tree.write(outfile, pretty_print=True)


'''  Insert xml tags into text based on spans
'''
def insert_tags(text, tags):
    print('insert_tags:', len(tags))
    lastindex = 0
    new_text = ""
    for tag in tags:
        if tag.tag == 'TLINK':
            new_text = new_text + etree.tostring(tag, encoding='utf8').decode('utf-8')
        else:
            #print(etree.tostring(tag, encoding='utf8').decode('utf-8'))
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
            # Fix escaped characters and read the node in properly as xml
            narr_string = escape_and(unescape(etree.tostring(narr_node, encoding='utf-8', with_tail=True).decode('utf-8')))
            #print('narr_string', narr_string)
            try:
                narr_node = etree.fromstring(narr_string)
            except etree.XMLSyntaxError as e:
                narr_lines = narr_string.splitlines()
                line, column = e.position
                print('XMLSyntaxError at line', line, 'column', column, ':', narr_lines[int(line)-1])
                exit(1)

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
            #print("tail:", tail)
            dct = unescape(etree.tostring(timex_node, encoding='utf-8', with_tail=False).decode('utf-8'))
            #print('DCT:', dct)
            #narr_node.remove(orig_node)

            # Remove empty event nodes
            for event_tag in narr_node.findall('EVENT'):
                #del event_tag.attrib['span'] # TEMP for ctakes
                #del event_tag.attrib['start'] # TEMP for ctakes
                if event_tag.text is None:
                    narr_node.remove(event_tag)
                    print('WARNING: Removed empty EVENT', event_tag.attrib['eid'], 'in', docid)
            # TEMP for ctakes
            #for time_tag in narr_node.findall('TIMEX3'):
            #    del event_tag.attrib['span'] # TEMP for ctakes
            #    del event_tag.attrib['start']

            narr_text = unescape(stringify_children(narr_node))
            if narr_text[0:4] == "None":
                narr_text = narr_text[4:]
            narr = narr_text # Don't include the DCT tail
            #narr = tail + narr_text

            #print("narr_text:", narr)

        else:
            print("ERROR: narr is None! Look for: ", node_name)
            #print("in child node:", etree.tostring(child))
            narr = ""
        # Create output tree
        root = etree.XML('<?xml version="1.0"?><TimeML xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://timeml.org/timeMLdocs/TimeML_1.2.1.xsd"></TimeML>')
        tree = etree.ElementTree(root)
        #docid_node = etree.SubElement(root, "DOCID")
        #docid_node.text = docid
        #dct_node = etree.SubElement(root, "DCT")
        #dct_node.text = dct
        #dct_node.append(timex_node)
        text_node = etree.SubElement(root, "TEXT")
        text_node.text = unescape(narr)
        filename = os.path.join(dirname, docid + ".tml")
        tree.write(filename, encoding='utf-8')

        # Fix arrows
        fix_arrows_only(filename)

        # Unsplit punctuation
        #unsplit_punc(filename)

def unsplit_punc(filename):
    subprocess.call(["sed", "-i", "-e", 's/ ,/,/g', filename])
    subprocess.call(["sed", "-i", "-e", 's/ \././g', filename])
    subprocess.call(["sed", "-i", "-e", "s/ '/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ :/:/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ - /-/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/\[ /[/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ \]/]/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ = /=/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/ \/ /\//g", filename])

    #subprocess.call(["sed", "-i", "-e", "s/\([1-9]\)-\([a-zA-Z]\)/\1 - \2/g", filename])

# Temp function for fixing anafora dirs
def fix_dirs():
    source_dir = '/u/sjeblee/research/data/thyme/temporal/Test'
    target_dir = '/u/sjeblee/research/data/thyme/anafora/test'
    text_dir = '/u/sjeblee/research/data/thyme/text/test'
    ref_dir = '/u/sjeblee/research/data/thyme/anafora/regenerated'

    for dir_name in os.listdir(ref_dir):
        new_dir = os.path.join(target_dir, dir_name)
        os.mkdir(new_dir)
        # Copy text file
        shutil.copyfile(os.path.join(text_dir, dir_name), os.path.join(new_dir, dir_name))
        # Copy anafora file
        filename = dir_name + '.Temporal-Relation.gold.completed.xml'
        shutil.copyfile(os.path.join(source_dir, filename), os.path.join(new_dir, filename))
