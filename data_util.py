#!/usr/bin/python
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
                                                    
