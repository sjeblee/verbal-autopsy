#!/usr/bin/python
# -*- coding: utf-8 -*-
# Generate useful info from heideltime tags and depparses

from lxml import etree
from lxml.etree import tostring
from itertools import chain
import argparse
import numpy
import string
import subprocess

class Word:
    def __init__(self, word, index):
        self.word = word
        self.index = index
    def __str__(self):
        return str(self.index) + ": " + self.word

class Link:
    def __init__(self, name, source, sindex, target, tindex):
        self.name = name
        self.source = source
        self.sindex = sindex
        self.target = target
        self.tindex = tindex

    def __str__(self):
        return self.name + ": " + str(self.sindex) + " " + self.source + " - " + str(self.tindex) + " " + self.target

class TempPhrase:
    def __init__(self, name, index):
        self.name = name.strip()
        self.startindex = index
        num = len(self.name.split(' '))
        print "num words in tp: " + str(num)
        self.endindex = index + (num-1)

    def __str__(self):
        return str(self.startindex) + "-" + str(self.endindex) + " " + self.name

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./link_heidel_parse.py --in [file_depparse_heidel.xml] --out [outfile.xml]"
        exit()

    global temp_start, temp_end
    temp_start = "<TIMEX3"
    temp_end = "</TIMEX3>"

    # Get the xml from file
    tree = etree.parse(args.infile)
    root = tree.getroot()
    
    for child in root:
        node = child.find("narrative")
        narr = ""
        narr_dp = ""
        narr_ht = ""
        if node != None:
            narr = node.text.encode('utf-8')
        if len(narr) > 0:
            # Setup
            sent_temps = [] # temporal phrases by sentence
            sent_words = [] # words by sentence (with tags removed)
            sent_deps = [] # dependency parse links by sentence

            # Get the dependency parse node
            node_dp = child.find("narr_depparse")
            if node_dp != None:
                narr_dp = node_dp.text.encode('utf-8')

            # Get the TimeML node
            node_ht = child.find("narr_heidel")
            if node_ht != None:
                node_timeml = node_ht.find("TimeML")
                narr_ht = stringify_children(node_timeml).encode('utf-8')
                print "narr_ht: " + narr_ht
                narr_ht_lines = narr_ht.split(' .')
                for line in narr_ht_lines:
                    words, temp_phrases = parse_timeml(line)
                    sent_temps.append(temp_phrases)
                    sent_words.append(words)

            # Get depparses
            sent_deps = get_deps(narr_dp)

            print "sent_temps: " + str(len(sent_temps))
            print str(sent_temps)
            for x in range(len(sent_temps)):
                temps = sent_temps[x]
                deps = sent_deps[x]
                print "sent: " + str(x)
                for tp in temps:
                    print "tp: " + str(tp)
                    phrase = ""
                    phrase_words = [] # array of Words
                    used_indices = []
                    # Keep track of which indices have already been added to the phrase
                    for ind in range(tp.startindex, tp.endindex+1):
                        used_indices.append(ind)
                    # TODO: also don't add words that are already part of the temporal phrase
                    for ind in range(tp.startindex, tp.endindex+1): 
                        print "checking index " + str(ind)
                        print "used_indices: " + str(used_indices)
                        for links in deps.values():
                            #print ind + " links: " + str(len(links))
                            for link in links:
                                cont = True
                                sind = int(link.sindex)
                                tind = int(link.tindex)
                                print "checking link: " + str(sind) + ":" + str(tind)
                                #print "sind==ind " + str(sind==ind)
                                #print "tind==ind " + str(tind==ind)
                                
                                if sind == ind and tind not in used_indices:
                                    print "-- " + str(link)
                                    word = Word(link.target, tind)
                                    phrase_words.append(word)
                                    used_indices.append(tind)
                                    # Follow forward links TODO: fix this
                                    for link2 in links:
                                        if int(link2.sindex) == tind:
                                            print "--- " + str(link2)
                                            if link.name in ["prep","xcomp","pobj","conj","cc"]:
                                                phrase_words.append(word)
                                elif tind == ind and sind not in used_indices:
                                    print "-- " + str(link)
                                    word = Word(link.source, sind)
                                    phrase_words.append(word)
                                    used_indices.append(sind)
                                    for link2 in links:
                                        if int(link2.tindex) == sind:
                                            print "--- " + str(link2)
                                            #if link.name in ["prep","xcomp","pobj","conj","cc"]:
                                            phrase_words.append(word)

                    # TODO: construct phrase in the correct order
                    phrase_words.sort(key=lambda x: x.index, reverse=False)
                    for w in phrase_words:
                        phrase = phrase + " " + w.word
                    phrase = phrase.strip()
                    print "link phrase: " + phrase
            # TODO: find the head of the temporal phrase in the dependency parse
            # TODO: find the link from the temporal phrase head to another phrase

            #child.append(newnode)
        
    # write the new xml to file
    tree.write(args.outfile)

def get_deps(narr_dp):
    links = {}
    sent_deps = []
    for line in narr_dp.split('\n'):
        line = line.strip()
        # Store the links of the parse in a dictionary
        if len(line) == 0:
            sent_deps.append(links)
            links = {}
        else:
            i = line.index('(')
            name = line[0:i]
            j = line.index(")")
            stuff = line[i+1:j]
            words = stuff.split(', ')
            k = words[0].index('-')
            source = words[0][0:k]
            sindex = words[0][k+1:]
            l = words[1].index('-')
            target = words[1][0:l]
            tindex = words[1][l+1:]
            link = Link(name, source, sindex, target, tindex)
            if not links.has_key(source):
                links[sindex] = []
            links[sindex].append(link)
            print "added link: " + str(link)

    print "sent_deps: " + str(len(sent_deps))
    print str(sent_deps)
    return sent_deps

def parse_timeml(line):
    line = line.strip()
    print "parsing: " + line
    words = []
    tps = []
    words.append("ROOT")
    in_tp = False
    start = -1
    ws = line.split(' ')
    index = 0
    temp_text = ""
    while index < len(ws):
        w = ws[index]
        if in_tp:
            if temp_end in w:
                # Close temporal tag
                tp = TempPhrase(temp_text, start)
                print "adding TP: " + str(tp)
                tps.append(tp)
                in_tp = False
                temp_text = ""
            else:
                temp_text = temp_text + " " + w
                words.append(w)
        elif temp_start in w:
            in_tp = True
            start = len(words)
            # TODO: get type and value???
            while ">" not in w:
                index = index+1
                w = ws[index]
        else:
            words.append(w)
        index = index + 1

    print "words: " + str(words)
    return words, tps

def stringify_children(node):
    parts = ([node.text] + list(chain(*([tostring(c)] for c in node.getchildren()))))
    for part in parts:
        print "part: " + part
    # filter removes possible Nones in texts and tails
    return ''.join(filter(None, parts))

if __name__ == "__main__":main()
