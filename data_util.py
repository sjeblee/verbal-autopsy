#!/usr/bin/python
# -*- coding: utf-8 -*-
# Util functions
# @author sjeblee@cs.toronto.edu

from lxml import etree
from lxml.etree import tostring
from itertools import chain
from sklearn import metrics
import numpy
import operator
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

def decode_ageunit(unit):
    unit = str(unit).lower()
    if unit == "3" or "year" in unit:
        return "years"
    elif unit == "2" or "month" in unit:
        return "months"
    elif unit == "1" or "day" in unit:
        return "days"
    else:
        return None

def encode_ageunit(unit):
    unit = unit.lower()
    if unit == "years":
        return "3"
    elif unit == "months":
        return "2"
    elif unit == "days":
        return "1"
    else:
        return None

def decode_gender(gender):
    gender = str(gender).lower()
    if gender == "1" or gender == "male":
        return "male"
    elif gender == "2" or gender == "female":
        return "female"
    else:
        return None

def encode_gender(gender):
    gender = gender.lower()
    if gender == "male":
        return "1"
    elif gender == "female":
        return "2"
    else:
        return None

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

''' Labels must be integers or the empty string!
    labels: [num_samples, seq_len]
    returns: [num_samples, num_classes]
'''
def multi_hot_encoding(labels, max_label=None):
    encoded_labels = []

    # Figure out the dimensionality
    if max_label is None:
        max_label = 0
        for seq in labels:
            for item in seq:
                if item != "":
                    val = int(item)
                    if val > max_label:
                        max_label = val
    dim = max_label+1
    print "max_label: " + str(max_label) + ", dim: " + str(dim)

    for seq in labels:
        encoded_seq = zero_vec(dim)
        for item in seq:
            if item != "":
                num = int(item)
                encoded_seq[num] = 1
        encoded_labels.append(encoded_seq)
    return encoded_labels

''' Convert multi-hot labels back to a text list of cluster numbers
'''
def decode_multi_hot(labels):
    decoded_labels = []
    for lab in labels:
        label_seq = ""
        for x in range(len(lab)):
            val = lab[x]
            if val >= 1:
                label_seq = label_seq + "," + str(x)
        decoded_labels.append(label_seq.strip(','))
    return decoded_labels

''' labels: [num_samples, num_clusters]
    returns: a list of multi-hot vectors
'''
def map_to_multi_hot(labels, threshold=0.1):
    decoded_labels = []
    dim = len(labels[0])
    for lab in labels:
        label_seq = zero_vec(dim)
        for x in range(len(lab)):
            val = lab[x]
            if val >= threshold:
                label_seq[x] = 1
        decoded_labels.append(label_seq)
    return decoded_labels

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

def score_majority_class(true_labs):
    pred_labs = []
    majority_lab = None
    count_map = {}
    for lab in true_labs:
        if lab not in count_map.keys():
            count_map[lab] = 0
        count_map[lab] = count_map[lab]+1
    majority_lab = max(count_map.iteritems(), key=operator.itemgetter(1))[0]
    for lab in true_labs:
        pred_labs = majority_lab
    # Score
    precision = metrics.precision_score(true_labs, pred_labs, average="weighted")
    recall = metrics.recall_score(true_labs, pred_labs, average="weighted")
    f1 = metrics.f1_score(true_labs, pred_labs, average="weighted")
    return precision, recall, f1

''' Scores vector labels with binary values
    returns: avg precision, recall, f1 of 1 labels (not 0s)
'''
def score_vec_labels(true_labs, pred_labs):
    p_scores = []
    r_scores = []
    f1_scores = []
    micro_pos = 0
    micro_tp = 0
    micro_fp = 0
    assert(len(true_labs) == len(pred_labs))
    for x in range(len(true_labs)):
        true_lab = true_labs[x]
        pred_lab = pred_labs[x]
        pos = 0
        tp = 0
        fp = 0
        for y in range(len(true_lab)):
            true_val = true_lab[y]
            pred_val = pred_lab[y]
            if true_val == 1:
                pos = pos+1
                micro_pos = micro_pos+1
                if pred_val == 1:
                    tp = tp+1
                    micro_tp=micro_tp+1
            else:
                if pred_val == 1:
                    fp = fp+1
                    micro_fp = micro_fp+1

        p = 0.0
        r = 0.0
        if (tp+fp) > 0:
            p = float(tp) / float(tp+fp)
        if pos > 0:
            r = float(tp) / float(pos)
        if p == 0.0 and r == 0.0:
            f1 = float(0)
        else:
            f1 = 2*(p*r)/(p+r)
        p_scores.append(p)
        r_scores.append(r)
        f1_scores.append(f1)
    precision = numpy.average(p_scores)
    recall = numpy.average(r_scores)
    f1 = numpy.average(f1_scores)
    micro_p = 0.0
    micro_r = 0.0
    if (micro_tp+micro_fp) > 0:
        micro_p = float(micro_tp) / float(micro_tp+micro_fp)
    if micro_pos > 0:
        micro_r = float(micro_tp) / float(micro_pos)
    if micro_p == 0.0 and micro_r == 0.0:
        micro_f1 = float(0)
    else:
        micro_f1 = 2*(micro_p*micro_r)/(micro_p+micro_r)
    return precision, recall, f1, micro_p, micro_r, micro_f1

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

def xml_to_txt(filename):
    name = filename.split(".")[0]
    sed_command = r"sed '$d' < " + filename + r" | sed '1d' > " + name + ".txt"
    ps = subprocess.Popen(sed_command, shell=True, stdout=subprocess.PIPE)
    ps.communicate()

def zero_vec(dim):
    vec = []
    for x in range(dim):
        vec.append(0)
    return vec
