#!/usr/bin/python3
# Assign a random category to each record
# @author sjeblee@cs.toronto.edu

import argparse
import numpy
import time
from lxml import etree

labelencoder = None
global adult_map, child_map, neonate_map

adult_map = ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '13', '14', '15', '16', '17']
child_map = ['1', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16']
neonate_map = ['1', '3', '5', '6', '8', '11', '13', '14', '15']

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--type', action="store", dest="type")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.type):
        print('usage: python model.py --in [test.xml] --out [test.results] --labels [ICD_cat/Final_code] -- type [adult/child/neonate]')
        exit()

    global labelname
    labelname = 'ICD_cat'
    if args.labelname:
        labelname = args.labelname

    run(args.infile, args.outfile, labelname, args.type)


def run(arg_test_file, arg_result_file, arg_labelname, arg_type):
    total_start_time = time.time()

    global labelname
    labelname = arg_labelname

    # Get the data
    testids, correctlabels = preprocess(arg_test_file)
    print('Y:', str(len(correctlabels)))
    testlabels = map_forward(correctlabels, arg_type)

    # Test
    results = test(len(testlabels), arg_type)
    predictedlabels = map_back(results, arg_type)
    #correctlabels = map_back(testlabels, arg_type)

    # Write results to a file
    if arg_result_file is not None:
        output = open(arg_result_file, 'w')
        for i in range(len(testids)):
            out = {}
            out['MG_ID'] = testids[i]
            out['Correct_ICD'] = correctlabels[i]
            out['Predicted_ICD'] = predictedlabels[i]
            output.write(str(out) + "\n")
        output.close()

    total_time = (time.time() - total_start_time) / 60
    print('total time:', str(total_time), 'mins')
    return correctlabels, predictedlabels


def test(num, rec_type):
    print('testing...')
    num_classes = len(adult_map)
    if rec_type == 'child':
        num_classes = len(child_map)
    elif rec_type == 'neonate':
        num_classes = len(neonate_map)

    predictedlabels = numpy.random.randint(0, num_classes, num)
    #print "testY: " + str(testY)
    #print "results: " + str(results)
    return predictedlabels


def preprocess(filename):
    # Get the ids and labels from the test file
    #starttime = time.time()
    ids = []
    labels = []
    root = etree.parse(filename).getroot()
    for child in root:
        node = child.find('MG_ID')
        val = node.text
        ids.append(val)
        item = child.find(labelname)
        value = '15'
        if item is not None:
            value = item.text
        labels.append(value)

    #endtime = time.time()
    #print "preprocessing took " + str(endtime - starttime) + " s"
    return ids, labels


def map_forward(labels, label_map):
    new_labels = []
    #label_map = adult_map
    #if rec_type == "child":
    #    label_map = child_map
    #elif rec_type == "neonate":
    #   label_map = neonate_map
    for y in labels:
        l = label_map.index(y)
        new_labels.append(l)
    return new_labels


def map_back(results, label_map):
    #print "map_back: " + rec_type + " " + str(len(results))
    new_labels = []
    #label_map = adult_map
    #if rec_type == "child":
    #    label_map = child_map
    #elif rec_type == "neonate":
    #    label_map = neonate_map
    #print "label_map: " + str(label_map)
    for y in results:
        #print "mapping result: " + str(y)
        l = label_map[int(y)]
        new_labels.append(l)
    return new_labels


if __name__ == "__main__": main()
