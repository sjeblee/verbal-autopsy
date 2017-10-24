#!/usr/bin/python
# -*- coding: utf-8 -*-
# Cluster the keywords from the records using word2vec

from lxml import etree
from keras.models import Sequential, load_model
from keras.layers import Dense, Activation
from keras.utils.np_utils import to_categorical
from sklearn.cluster import KMeans, SpectralClustering
from scipy.stats import mode
import argparse
import numpy
import time

import cluster_keywords
import extract_features

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--vectors', action="store", dest="vecfile")
    args = argparser.parse_args()

    if not (args.trainfile and args.testfile and args.vecfile and args.outfile):
        print "usage: ./cluster_keywords.py --train [file.csv] --test [file.csv] --out [file.txt] --vectors [file.vectors]"
        exit()

    run(args.trainfile, args.testfile, args.outfile, args.vecfile)

def run(trainfile, testfile, outfile, vecfile):
    starttime = time.time()

    # Load word2vec vectors
    print "loading vectors..."
    word2vec, dim = extract_features.load_word2vec(vecfile)
    
    # Extract keywords
    train_keywords, train_clusters, train_vecs, cluster_names = cluster_keywords.read_cluster_file(trainfile, word2vec, dim)
    test_keywords, test_clusters, test_vecs, test_cluster_names = cluster_keywords.read_cluster_file(testfile, word2vec, dim, cluster_names)
    num_clusters = len(cluster_names)
    
    print "train_keywords: " + str(len(train_keywords))
    print "num_clusters: " + str(num_clusters)
    print "dim: " + str(dim)

    # Generate clusters
    print "generating clusters..."
    nn = Sequential([Dense(200, input_dim=dim),
                     Activation('relu'),
                     #Dense(num_nodes, input_dim=num_feats),
                     #Activation(activation),
                     Dense(num_clusters),
                     Activation('softmax'),])

    nn.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    print "train shape: vecs: " + str(numpy.array(train_vecs).shape) + "clusters: " + str(numpy.array(train_clusters).shape)
    nn.fit(numpy.array(train_vecs), to_categorical(train_clusters))
    nn.summary()
                
    results = nn.predict(numpy.array(test_vecs))
    pred_clusters = map_back(results, cluster_names)

    # Score clusters
    print "scoring clusters..."
    purity_score = cluster_keywords.purity(test_keywords, test_clusters, pred_clusters)
    print "purity: " + str(purity_score)

    # Write results to file
    cluster_keywords.write_clusters_to_file(outfile, cluster_keywords.get_cluster_map(test_keywords, pred_clusters))

    totaltime = time.time() - starttime
    print "Total time: " + str(totaltime) + " s"

def map_back(clusters, cluster_names):
    cluster_vals = []
    for c_vec in clusters:
        val = numpy.argmax(c_vec)
        cluster_vals.append(cluster_names[val])
    return cluster_vals
                                
    
if __name__ == "__main__":main()
