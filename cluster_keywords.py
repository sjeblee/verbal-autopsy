#!/usr/bin/python
# -*- coding: utf-8 -*-
# Cluster the keywords from the records using word2vec

from lxml import etree
from sklearn.cluster import KMeans, SpectralClustering
from scipy.stats import mode
import argparse
import numpy
import time

import extract_features

def main():
    argparser = argparse.ArgumentParser()
    #argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--clusters', action="store", dest="clusterfile")
    argparser.add_argument('--vectors', action="store", dest="vecfile")
    args = argparser.parse_args()

    if not (args.outfile and args.clusterfile and args.vecfile):
        print "usage: ./cluster_keywords.py --out [file.txt] --clusters [file.csv] --vecfile [file.vectors]"
        exit()

    run(args.outfile, args.clusterfile, args.vecfile)

def run(outfile, clusterfile, vecfile):
    starttime = time.time()

    stopwords = ["a", "about", "above", "after", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "between", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "during", "each", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "him", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

    # Load word2vec vectors
    print "loading vectors..."
    word2vec, dim = extract_features.load_word2vec(vecfile)
    zero_vec = []
    for x in range(dim):
        zero_vec.append(0)
    zero_vec = numpy.array(zero_vec)
    
    # Extract keywords
    keywords = []
    kw_vecs = []
    kw_clusters_correct = []
    cluster_names = set()

    # Get the xml from file
    #root = etree.parse(infile).getroot()
    #for child in root:
    #    kws = extract_features.get_keywords(child)
    #    for kw in kws:
    print "reading cluster file..."
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
            kw_clusters_correct.append(clust)
            cluster_names.add(clust)
    num_clusters = len(cluster_names)
    print "num_keywords: " + str(len(keywords))
    print "num_clusters: " + str(num_clusters)
    print "dim: " + str(dim)

    # Generate clusters
    print "generating clusters..."
    clusterer = KMeans(n_clusters=num_clusters, n_jobs=1, precompute_distances=False, max_iter=300, n_init=15)
    #clusterer = SpectralClustering(n_clusters=num_clusters, n_init=15)
    kw_clusters = clusterer.fit_predict(kw_vecs)

    # Label clusters
    kw_clusters_predicted = []
    pred_cluster_map = {}
    for x in range(len(keywords)):
        kw = keywords[x]
        clust = kw_clusters[x]
        #clust_corr = kw_clusters_correct[x]
        if clust not in pred_cluster_map:
            pred_cluster_map[clust] = []
        pred_cluster_map[clust].append(kw)

    # Score clusters
    print "scoring clusters..."
    n = 0
    n_corr = 0
    for key in pred_cluster_map:
        kws = pred_cluster_map[key]
        correct_labels = []
        for kw in kws:
            lab = kw_clusters_correct[keywords.index(kw)]
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
    print "purity: " + str(purity)

    # Write results to file
    outf = open(outfile, 'w')
    for key in pred_cluster_map:
        outf.write(str(key) + ",")
        for item in pred_cluster_map[key]:
            outf.write(item + ",")
        outf.write("\n")
    outf.close()

    totaltime = time.time() - starttime
    print "Total time: " + str(totaltime) + " s"
    
if __name__ == "__main__":main()
