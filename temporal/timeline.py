#!/usr/bin/python
# -*- coding: utf-8 -*-
# Create a timeline from tagged text

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util

from lxml import etree
import argparse
import numpy

unk = "UNK"

class Event:
    text = ""
    time = unk
    neg = False

    def __init__(self, text, time, neg=False):
        self.text = text
        self.time = time
        self.neg = neg

    def __str__(self):
        return self.time + ' : ' + self.text()

    def text(self):
        if self.neg:
            return 'no ' + self.text
        else:
            return self.text

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--in', action="store", dest="infile")
    argparser.add_argument('-o', '--out', action="store", dest="outfile")
    argparser.add_argument('-v', '--vectors', action="store", dest="vecfile")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.vecfile):
        print "usage: ./timeline.py --in [file_timeml.xml] --out [out.features.timeline] --vectors [vecfile.bin]"
        exit()

    run(args.infile, args.outfile, args.vecfile)

def run(infile, outfile, vecfile):
    # Get the xml from file
    tree = etree.parse(infile)
    root = tree.getroot()

    timelines = []
    for child in root:
        node = child.find("narr_timeml_simple")
        narr = ""
        if node != None:
            narr = node.text.encode('utf-8')
            timeline, events = create_timeline(narr, vecfile)
            timelines.append(events)

    out = open(outfile, 'w')
    out.write(str(timelines))
    out.close()

def create_timeline(text, vecfile):
    events = get_events(text)
    ordered_events = order_events(events) # TODO
    timeline = represent_events(ordered_events, vecfile)
    return timeline, ordered_events

''' Get event objects from TimeML tagged text
    text: text with TimeML event and timex3 tags
    returns: a list of Event objects
'''
def get_events(text):
    events = []
    tags = ['EVENT', 'TIMEX3']
    event_atts = ['polarity', 'relatedToTime']
    time_atts = ['id', 'type','val']
    event_phrases = data_util.phrases_from_tags(text, ['EVENT'], event_atts)
    time_phrases = data_util.phrases_from_tags(text, ['TIMEX3'], time_atts)
    times = {}
    for t in time_phrases:
        tid = t['id']
        times[tid] = t
    for ev in event_phrases:
        # Construct an Event object
        neg = False
        phrase = ev['text'].strip()
        pol = ev['polarity']
        if pol != None and pol == 'neg':
            neg = True
        tid = ev['relatedToTime']
        if tid is None or tid == '':
            time = unk
        else:
            time_phrase = times[tid]
            # TODO: get actual time value from time elements
            time = time_phrase['text'].strip()
        event = Event(phrase, time, neg)
        events.append(event)
    return events

def order_events(events):
    # TODO: order events by time value and then by order of mention in the text
    print "Events not ordered! (TODO)"
    return events

def represent_events(events, vecfile):
    global zero_vec
    zero_vec = []

    # Load vectors
    word2vec, dim = data_util.load_word2vec(vecfile)
    for z in range(0, dim):
        zero_vec.append(0)
    vectors = []
    for event in events:
        phrase = get_avg_vec(event.text(), word2vec)
        time_vec = zero_vec
        if event.time != unk:
            time_vec = get_avg_vec(event.time, word2vec)
        vector = time_vec + phrase
        vectors.append(vector)
    return vectors

def get_avg_vec(text, word2vec):
    vecs = []
    for word in text.split(' '):
        vec = zero_vec
        if word in word2vec:
            vec = word2vec[word]
        vecs.append(vec)
    avg_vec = numpy.average(vecs, axis=0)
    return avg_vec

if __name__ == "__main__":main()
