#!/usr/bin/python
# -*- coding: utf-8 -*-
# Cluster the keywords from the records using word2vec

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util

from collections import Counter
from lxml import etree
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import argparse
import numpy
import time

import extract_features_temp as extract_features

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--clusters', action="store", dest="clusterfile")
    argparser.add_argument('--vectors', action="store", dest="vecfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--testout', action="store", dest="testoutfile")
    argparser.add_argument('-n', '--num', action="store", dest="num") # number of clusters
    args = argparser.parse_args()

    if not (args.outfile and args.clusterfile and args.vecfile):
        print "usage: ./cluster_keywords.py --in [file.xml] --out [file.xml] --clusters [file.csv] --vecfile [file.vectors] --test [file.xml] --testout [file.xml] --num [n_clusters]"
        exit()

    num = 20
    if args.num:
        num = int(args.num)

    if args.testfile and args.infile:
        run(args.outfile, args.clusterfile, args.vecfile, args.infile, args.testfile, args.testoutfile, num)
    if args.infile:
        run(args.outfile, args.clusterfile, args.vecfile, args.infile, num_clusters=num)
    else:
        run(args.outfile, args.clusterfile, args.vecfile, num_clusters=num)

def run(outfile, clusterfile, vecfile, infile=None, testfile=None, testoutfile=None, num_clusters=20):
    starttime = time.time()

    #stopwords = ["a", "about", "above", "after", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "between", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "during", "each", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "him", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

    # Load word2vec vectors
    print "loading vectors..."
    word2vec, dim = extract_features.load_word2vec(vecfile)
    
    # Extract keywords
    ids = [] # ids can occur more than once!
    keywords = []
    kw_vecs = []
    kw_clusters_correct = []
    cluster_names = []

    if infile is not None:
        print "reading XML file..."
        # Get the xml from file
        ids, keywords, kw_vecs = read_xml_file(infile, word2vec, dim)
    else:
        print "reading cluster file..."
        keywords, kw_clusters_correct, kw_vecs, cluster_names = read_cluster_file(clusterfile, word2vec, dim)
        num_clusters = len(cluster_names)
    print "num_keywords: " + str(len(keywords))
    print "num_clusters: " + str(num_clusters)
    print "dim: " + str(dim)
    print "cluster_names: " + str(cluster_names)

    # Generate clusters
    print "shape: [num_keywords, dim]" # keywords listed individually
    print "generating clusters..."
    #clusterer = KMeans(n_clusters=num_clusters, n_jobs=1, precompute_distances=False, max_iter=500, n_init=15)
    #clusterer = SpectralClustering(n_clusters=num_clusters, n_init=15, affinity='nearest_neighbors')
    clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
    kw_clusters = clusterer.fit_predict(kw_vecs)
    #kw_clusters = map_back(clusterer.fit_predict(kw_vecs), cluster_names)

    # Test
    if testfile is not None:
        classifier = KNeighborsClassifier(algorithm='ball_tree')
        classifier.fit(kw_vecs, kw_clusters)
        if ',' in testfile:
            testfiles = testfile.split(',')
            testoutfiles = testoutfile.split(',')
            testnames = zip(testfiles, testoutfiles)
            for filename, outfilename in testnames:
                print "predicting clusters for test file: " + filename
                testids, testkeywords, testvecs = read_xml_file(filename, word2vec, dim)
                test_clusters = classifier.predict(testvecs)
                write_clusters_to_xml(filename, outfilename, testids, test_clusters)
        else:
            print "predicting clusters for test file..."
            testids, testkeywords, testvecs = read_xml_file(testfile, word2vec, dim)
            test_clusters = classifier.predict(testvecs)
            write_clusters_to_xml(testfile, testoutfile, testids, test_clusters)

    # Score clusters
   # print "scoring clusters..."
   # purity_score = purity(keywords, kw_clusters_correct, kw_clusters)
   # print "purity: " + str(purity_score)

    # Write results to file
    print "Adding cluster labels to xml tree..."
    write_clusters_to_xml(infile, outfile, ids, kw_clusters)
    print "Writing clusters to csv file..."
    write_clusters_to_file(clusterfile, get_cluster_map(keywords, kw_clusters))
    #outf = open(outfile + ".vecs", 'w')
    #outf.write(str(get_cluster_map(kw_vecs, kw_clusters_correct, cluster_names)))
    #outf.close()

    totaltime = time.time() - starttime
    print "Total time: " + str(totaltime) + " s"

def read_cluster_file(clusterfile, word2vec, dim, cluster_names=None):
    train = False
    if cluster_names is None:
        cluster_names = set()
        cluster_names.add(0)
        train = True
    keywords = []
    kw_vecs = []
    kw_clusters = []

    zero_vec = []
    for x in range(dim):
        zero_vec.append(0)
    zero_vec = numpy.array(zero_vec)

    with open(clusterfile, 'r') as f:
            for line in f:
                cols = line.split(',')
                kw = cols[0]
                clust = int(cols[1].strip())
                # Look up keyword in word2vec
                vec = zero_vec
                for word in kw.split(' '):
                    vec2 = zero_vec
                    # ignore stopwords
                    #if word not in stopwords and word in word2vec:
                    if word in word2vec:
                        vec2 = numpy.array(word2vec[word])
                    vec = vec+vec2
                keywords.append(kw)
                kw_vecs.append(vec)
                kw_clusters.append(clust)
                if train:
                    cluster_names.add(clust)

    # Convert cluster names to numbers 0 to num_clusters
    if train:
        cluster_names = list(cluster_names)
    for x in range(len(kw_clusters)):
        val = kw_clusters[x]
        if val in cluster_names:
            kw_clusters[x] = cluster_names.index(val)
        else:
            kw_clusters[x] = 0

    return keywords, kw_clusters, kw_vecs, cluster_names

def read_xml_file(filename, word2vec, dim):
    ids = []
    keywords = []
    kw_vecs = []
    root = etree.parse(filename).getroot()
    for child in root:
        node = child.find('MG_ID')
        rec_id = node.text
        kws = extract_features.get_keywords(child).split(',')
        for kw in kws:
            kw = kw.strip()
            if len(kw) > 0:
                print rec_id + " : " + kw
                vec = vectorize(kw, word2vec, dim)
                if vec is not None:
                    ids.append(rec_id)
                    keywords.append(kw)
                    kw_vecs.append(vec)
                else:
                    print "DROPPED"
    return ids, keywords, kw_vecs

def purity(keywords, corr_clusters, pred_clusters):
    # Label clusters
    pred_cluster_map = get_cluster_map(keywords, pred_clusters)

    n = 0
    n_corr = 0
    for key in pred_cluster_map:
        kws = pred_cluster_map[key]
        correct_labels = []
        for kw in kws:
            lab = corr_clusters[keywords.index(kw)]
            correct_labels.append(lab)
        print "correct_labels: " + str(correct_labels)
        m, count = mode(numpy.array(correct_labels))
        label = m[0]
        print "label: " + str(label)
        for item in correct_labels:
            n = n+1
            if item == label:
                n_corr = n_corr+1

    # Calculate purity score
    print "n: " + str(n)
    print "n_corr: " + str(n_corr)
    purity = float(n_corr) / float(n)
    return purity

def get_cluster_map(keywords, clusters, cluster_names=None):
    cluster_map = {}
    for x in range(len(keywords)):
        kw = keywords[x]
        clust = clusters[x]
        if cluster_names is not None:
            clust = cluster_names[clust]
        if clust not in cluster_map:
            cluster_map[clust] = []
        cluster_map[clust].append(kw)
    return cluster_map

def write_clusters_to_file(outfile, cluster_map):
    print "writing clusters to " + outfile
    outf = open(outfile, 'w')
    for key in cluster_map:
        print "cluster " + str(key)
        outf.write(str(key) + ",")
        for item in cluster_map[key]:
            outf.write(str(item) + ",")
        outf.write("\n")
    outf.close()

def write_clusters_to_xml(xmlfile, outfile, ids, cluster_pred, kw_label="keyword_clusters"):
    # Create dictionary
    id_dict = {}
    for x in range(len(ids)):
        rec_id = ids[x]
        cluster = cluster_pred[x]
        if rec_id not in id_dict:
            id_dict[rec_id] = []
        id_dict[rec_id].append(cluster)
    # Read xml file and add attributes
    tree = etree.parse(xmlfile)
    root = tree.getroot()
    for child in root:
        node = child.find('MG_ID')
        rec_id = node.text
        keyword_text = ""
        if rec_id in id_dict:
            keywords = id_dict[rec_id]
            for kw in keywords:
                keyword_text = keyword_text + "," + str(kw)
        newnode = etree.SubElement(child, kw_label)
        newnode.text = keyword_text.strip(',')
    # Write tree to file
    tree.write(outfile)

def map_back(clusters, cluster_names):
    cluster_vals = []
    for val in clusters:
        cluster_vals.append(cluster_names[val])
    return cluster_vals

def vectorize(phrase, word2vec, dim):
    words = phrase.split(' ')
    vecs = []
    #zero_vec = data_util.zero_vec(dim)
    for word in words:
        if word in word2vec:
            vecs.append(word2vec.get(word))
        #else:
        #    vecs.append(zero_vec)
    # Average vectors
    if len(vecs) > 0:
        avg_vec = numpy.average(numpy.asarray(vecs), axis=0)
        return avg_vec
    else:
        return None

''' Converts cluster numbers to names
    clusters: a list of lists of cluster numbers
    clusterfile: the file containing the dictionary of cluster numbers to all the cluster text
    returns: a list of lists of cluster names
'''
def interpret_clusters(clusters, clusterfile):
    # Read clusters from file
    cluster_names = {}
    cluster_map = {}
    clusters_text = []
    with open(clusterfile, 'r') as f:
        for line in f.readlines():
            phrases = line.strip().strip(',').split(',')
            key = int(phrases[0])
            phrases = phrases[1:]
            cluster_map[key] = phrases
            label = Counter(phrases).most_common(1)[0][0]
            cluster_names[key] = label

    # Write cluster name mapping to file
    outname = clusterfile + ".names"
    outfile = open(outname, 'w')
    outfile.write(str(cluster_names))
    outfile.close()

    # Replace cluster names with text
    for entry in clusters:
        text = []
        for num in entry.split(','):
            if num is not '':
                name = cluster_names[int(num)]
                text.append(name)
        clusters_text.append(text)
    return clusters_text
    
if __name__ == "__main__":main()
