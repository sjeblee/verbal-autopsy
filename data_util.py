#!/usr/bin/python
# -*- coding: utf-8 -*-
# Util functions
# @author sjeblee@cs.toronto.edu

from lxml import etree
from lxml.etree import tostring
from itertools import chain
import subprocess

def clean_file(filename):
    # remove blank lines | remove extra spaces| remove leading and trailing spaces  | fix utf-8 chars
    command = r"sed '/^\s*$/d' $file | sed -e 's/  */ /g' | sed -e 's/^ //g' | sed -e 's/ $//g' | sed -e 's/&amp;/and/g' | sed -e 's/&#13;/ /g' | sed -e 's/&#8217;/\'/g' | sed -e 's/&#8221;/\"/g' | sed -e 's/&#8220;/\"/g' | sed -e 's/&#65533;//g' | sed -e 's/&#175\7;//g'| sed -e 's/&#1770;/\'/g'"
    # TODO

''' Convert arrows in text to non-arrows (for xml processing)
    filename: the file to fix (file will be overwritten)
'''
def fix_arrows(filename):
    sed_command = r"sed -e 's/-->/to/g' " + filename + r" | sed -e 's/->/to/g' | sed -e 's/ < / lt /g' | sed -e 's/ > / gt /g'"
    print "sed_command: " + sed_command
    #f = open("temp", 'wb')
    ps = subprocess.Popen(sed_command, shell=True, stdout=subprocess.PIPE)
    output = ps.communicate()[0]
    out = open(filename, 'w')
    out.write(output)
    out.close()

def fix_escaped_chars(filename):
    subprocess.call(["sed", "-i", "-e", 's/&lt;/ </g', filename])
    subprocess.call(["sed", "-i", "-e", 's/&gt;/> /g', filename])
    subprocess.call(["sed", "-i", "-e", 's/  / /g', filename])
    subprocess.call(["sed", "-i", "-e", "s/‘/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/’/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/&#8216;/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/&#8217;/'/g", filename])
    subprocess.call(["sed", "-i", "-e", "s/&#8211;/,/g", filename])

''' Remove blank lines, convert \n to space, remove double spaces, insert a line break before each record
    filename: the file to fix (file will be overwritten)
    rec_type: the type of record: adult, child, or neonate
'''
def fix_line_breaks(filename, rec_type):
    tag = "<Adult_Anonymous>"
    if rec_type == "child":
        tag = "<Child_Anonymous>"
    elif rec_type == "neonate":
        tag = "<Neonate_Anonymous>"
    sed_command = "s/" + tag + r"/\n" + tag + "/g"
    sed_command2 = r"sed -e 's/<\/root>/\n<\/root>/g'"
    #print "sed_command: " + sed_command
    tr_command = "tr " + r"'\n' " + "' '"
    #print "tr_command: " + tr_command
    #f = open("temp", 'wb')
    command = "sed -e '/^\s$/d' " + filename + " | " + tr_command + " | sed -e 's/  / /g' | sed -e '" + sed_command + "'" + " | " + sed_command2
    ps = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
    output = ps.communicate()[0]
    out = open(filename, 'w')
    out.write(output)
    out.close()

''' Read an icd category mapping from a csv file into a python dictionary
    filename: the name of the csv file (lines formatted as code,cat)
'''
def get_icd_map(filename):
    icd_map = {}
    with open(filename, "r") as f:
        for line in f:
            line = line.strip()
            parts = line.split(',')
            icd_map[parts[0]] = parts[1]
    return icd_map

def load_word2vec(vecfile):
    # Create word2vec mapping
    word2vec = {}
    dim = 0
    with open(vecfile, "r") as f:
        firstline = True
        for line in f:
            # Ignore the first line of the file
            if firstline:
                firstline = False
            else:
                tokens = line.strip().split(' ')
                vec = []
                word = tokens[0]
                for token in tokens[1:]:
                    vec.append(float(token))
                word2vec[word] = vec
                dim = len(vec)
    return word2vec, dim

def remove_no_narrs(infile, outfile):
    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()
    count = 0

    for child in root:
        node = child.find("narrative")
        if node == None:
            root.remove(child)
            count = count+1
        else:
            narr = node.text
            if narr == None or narr == "":
                root.remove(child)
                count = count+1

    print "Removed " + str(count) + " missing or empty narratives"
    tree.write(outfile)

''' Get content of a tree node as a string
    node: etree.Element
'''
def stringify_children(node):
    parts = ([node.text] + list(chain(*([tostring(c)] for c in node.getchildren()))))
    # filter removes possible Nones in texts and tails
    return ''.join(filter(None, parts))

''' Get contents of tags as a list of strings
    text: the xml-tagged text to process
    tags: a list of the tags to extract
    atts: a list of attributes to extract as well
'''
def phrases_from_tags(text, tags, atts=[]):
    for x in range(len(tags)):
        tags[x] = tags[x].lower()
    text = "<root>" + text + "</root>"
    phrases = []
    root = etree.fromstring(text)
    #print "phrases_from tags text: " + text
    for child in root:
        if child.tag.lower() in tags:
            print "found tag: " + child.tag
            phrase = {}
            if child.text != None:
                phrase['text'] = child.text
            for att in atts:
                if att in child.keys():
                    phrase[att] = child.get(att)
            phrases.append(phrase)
    return phrases

''' Get contents of tags as a list of strings
    text: the xml-tagged text to process
    tags: a list of the tags to extract
'''
def text_from_tags(text, tags):
    for x in range(len(tags)):
        tags[x] = tags[x].lower()
    text = "<root>" + text + "</root>"
    newtext = ""
    root = etree.fromstring(text)
    print "text: " + text
    for child in root:
        print "--child"
        if child.tag.lower() in tags:
            print "found tag: " + child.tag
            if child.text != None:
                newtext = newtext + ' ' + child.text
    return newtext

''' matrix: a list of dictionaries
    dict_keys: a list of the dictionary keys
    outfile: the file to write to
'''
def write_to_file(matrix, dict_keys, outfile):
    # Write the features to file
    print "writing " + str(len(matrix)) + " feature vectors to file..."
    output = open(outfile, 'w')
    for feat in matrix:
        #print "ICD_cat: " + feat["ICD_cat"]
        feat_string = str(feat).replace('\n', '')
        output.write(feat_string + "\n")
    output.close()

    key_output = open(outfile + ".keys", "w")
    key_output.write(str(dict_keys))
    key_output.close()
    return dict_keys

def zero_vec(dim):
    vec = []
    for x in range(dim):
        vec.append(0)
    return vec
