#!/usr/bin/python
# -*- coding: utf-8 -*-
# Cluster the keywords from the records using word2vec

import sys
sys.path.append('/u/sjeblee/research/va/git/verbal-autopsy')
import data_util

from collections import Counter
from lxml import etree
from numpy import array
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, SpectralClustering
from sklearn.metrics import calinski_harabaz_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import argparse
import numpy
import os
import textrank
import time

import extract_features
import word2vec

numpy.set_printoptions(threshold=numpy.nan)

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

    num = 100
    if args.num:
        num = int(args.num)

    if args.testfile and args.infile:
        run(args.outfile, args.clusterfile, args.vecfile, args.infile, args.testfile, args.testoutfile, num)
    elif args.infile:
        run(args.outfile, args.clusterfile, args.vecfile, args.infile, num_clusters=num)
    else:
        run(args.outfile, args.clusterfile, args.vecfile, num_clusters=num)

def run(outfile, clusterfile, vecfile, infile=None, testfile=None, testoutfile=None, num_clusters=100):
    starttime = time.time()

    #stopwords = ["a", "about", "above", "after", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being", "between", "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "during", "each", "for", "from", "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "him", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such", "than", "that", "that's", "the", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves"]

    # Load word2vec vectors
    print "loading vectors..."
    vec_model, dim = word2vec.load(vecfile)
    
    # Extract keywords
    ids = [] # ids can occur more than once!
    keywords = []
    kw_vecs = []
    kw_clusters_correct = []
    cluster_names = []

    if infile is not None:
        print "reading XML file..."
        # Get the xml from file
        ids, keywords, kw_vecs = read_xml_file(infile, vec_model, dim)
    else:
        print "reading cluster file..."
        keywords, kw_clusters_correct, kw_vecs, cluster_names = read_cluster_file(clusterfile, vec_model, dim)
        num_clusters = len(cluster_names)
    print "num_keywords: " + str(len(keywords))
    print "num_clusters: " + str(num_clusters)
    print "dim: " + str(dim)
    #print "cluster_names: " + str(cluster_names)

    # Generate clusters
    kw_vecs_np = numpy.asarray(kw_vecs)
    print "shape: [num_keywords, dim]: " + str(kw_vecs_np.shape) # keywords listed individually
    print "kw_vecs[0]: " + str(kw_vecs[0])
    print "generating clusters..."
    clusterer = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000, n_init=100, batch_size=500, reassignment_ratio=0.03, max_no_improvement=50)
    #clusterer = SpectralClustering(n_clusters=num_clusters, n_init=15, affinity='cosine', n_jobs=-1)
    #clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
    kw_clusters = clusterer.fit_predict(kw_vecs_np)
    #kw_clusters = map_back(clusterer.fit_predict(kw_vecs), cluster_names)
    

    # Unsupervised cluster metrics
    cluster_labels = clusterer.labels_
    cluster_centers = list(clusterer.cluster_centers_)
    chi_score = calinski_harabaz_score(kw_vecs, cluster_labels)
    print "Calinski-Harabaz Index: " + str(chi_score)

    print "Writing clusters to csv file..."
    write_clusters_to_file(clusterfile, get_cluster_map(keywords, kw_clusters))
    # Save the cluster centers
    outf = open(clusterfile + ".centers", 'w')
    outf.write(str(cluster_centers))
    outf.close()

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
                testids, testkeywords, testvecs = read_xml_file(filename, vec_model, dim)
                test_clusters = classifier.predict(testvecs)
                testids_collapsed, clusters_collapsed = collapse(testids, test_clusters)
                test_emb, test_clusters_text = cluster_embeddings(clusters_collapsed, clusterfile, vecfile, return_phrases=True)
                write_clusters_to_xml(filename, outfilename, testids, clusters_collapsed, test_clusters_text)
        else:
            print "predicting clusters for test file..."
            testids, testkeywords, testvecs = read_xml_file(testfile, vec_model, dim)
            test_clusters = classifier.predict(testvecs)
            testids_collapsed, clusters_collapsed = collapse(testids, test_clusters)
            test_emb, test_clusters_text = cluster_embeddings(clusters_collapsed, clusterfile, vecfile, return_phrases=True)
            write_clusters_to_xml(testfile, testoutfile, testids_collapsed, clusters_collapsed, test_clusters_text)

    # Get text interpretation of clusters
    ids_collapsed, clusters_collapsed = collapse(ids, kw_clusters)
    kw_emb, kw_clusters_text = cluster_embeddings(clusters_collapsed, clusterfile, vecfile, return_phrases=True)

    # Write results to file
    print "Adding cluster labels to xml tree..."
    write_clusters_to_xml(infile, outfile, ids_collapsed, clusters_collapsed, kw_clusters_text)

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

    zero_vec = numpy.array(data_util.zero_vec(dim))

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

def read_xml_file(filename, vec_model, dim):
    print "read_xml_file: " + filename
    ids = []
    keywords = []
    kw_vecs = []
    root = etree.parse(filename).getroot()
    for child in root:
        node = child.find('MG_ID')
        rec_id = node.text
        kws = extract_features.get_keywords(child, "keywords_spell").split(',')
        for kw in kws:
            kw = kw.strip()
            if len(kw) > 0:
                #print rec_id + " : " + kw
                vec = vectorize(kw, vec_model, dim)
                if vec is not None and (x is True for x in numpy.isfinite(numpy.asarray(vec)).tolist()):
                    ids.append(rec_id)
                    keywords.append(kw)
                    kw_vecs.append(vec)
                #else:
                #    print "DROPPED: " + kw
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
        print "cluster " + str(key) + " : " + str(len(cluster_map[key])) + " keywords"
        outf.write(str(key) + ",")
        for item in cluster_map[key]:
            outf.write(str(item) + ",")
        outf.write("\n")
    outf.close()

''' xmlfile: the xml file to add key phrases to
    ids: the list of record ids (collapsed)
    cluster_pred: a list of lists of cluster numbers (collapsed)
    text_pred: a list of lists of strings representing the cluster names (collapsed)
'''
def write_clusters_to_xml(xmlfile, outfile, ids, cluster_pred, text_pred=None, kw_label="keyword_clusters"):
    # Create dictionary
    id_dict = {}
    text_dict = {}
    for x in range(len(ids)):
        rec_id = ids[x]
        id_dict[rec_id] = cluster_pred[x]
        text_dict[rec_id] = text_pred[x]

    # Read xml file and add attributes
    tree = etree.parse(xmlfile)
    root = tree.getroot()
    for child in root:
        node = child.find('MG_ID')
        rec_id = node.text
        narr_node = child.find('narrative')
        # Add textrank key phrases for comparison
        if narr_node is not None:
            narr = narr_node.text
            kw_textrank = textrank.extract_key_phrases(narr)
            #print "kw_textrank: " + str(kw_textrank)
            tr_node = etree.SubElement(child, 'textrank_keyphrases')
            tr_node.text = str(kw_textrank)
        keyword_text = ""
        if rec_id in id_dict:
            keywords = id_dict[rec_id]
            kw_text = text_dict[rec_id]
            for kw in keywords:
                keyword_text = keyword_text + "," + str(kw)
        newnode = etree.SubElement(child, kw_label)
        newnode.text = keyword_text.strip(',')
        newnode2 = etree.SubElement(child, kw_label + "_text")
        newnode2.text = str(kw_text)
    # Write tree to file
    tree.write(outfile)

def collapse(ids, clusters):
    ids_collapsed = []
    feats_collapsed = []
    id_dict = {}
    for x in range(len(ids)):
        rec_id = ids[x]
        cluster = clusters[x]
        if rec_id not in id_dict:
            id_dict[rec_id] = []
        id_dict[rec_id].append(cluster)
    ids_collapsed = id_dict.keys()
    for idc in ids_collapsed:
        feats_collapsed.append(id_dict[idc])
    return ids_collapsed, feats_collapsed

def map_back(clusters, cluster_names):
    cluster_vals = []
    for val in clusters:
        cluster_vals.append(cluster_names[val])
    return cluster_vals

def vectorize(phrase, vec_model, dim):
    stopwords = ['and', 'or', 'with', 'in', 'of', 'at', 'had', 'ho']
    words = phrase.split(' ')
    vecs = []
    zero_vec = numpy.asarray(data_util.zero_vec(dim))
    for word in words:
        if word not in stopwords:
            vecs.append(word2vec.get(word, vec_model))
    # Average vectors
    if len(vecs) > 0:
        avg_vec = numpy.average(numpy.asarray(vecs), axis=0)
        return avg_vec
    else:
        return zero_vec

def clusters_from_file(clusterfile):
    print "TODO"

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
            if num != '':
                if int(num) in cluster_names:
                    name = cluster_names[int(num)]
                    text.append(name)
                else:
                    print "NOT FOUND: " + num
        clusters_text.append(text)
    return clusters_text

''' Get cluster embeddings for keyword clusters
    labels: list of lists of cluster numbers
'''
def cluster_embeddings(labels, clusterfile, vecfile, return_phrases=False, max_length=10):
    print "getting cluster embeddings"
    word2vec, dim = data_util.load_word2vec(vecfile)
    cluster_map = {}
    cluster_names = {}
    cluster_embeddings = []
    with open(clusterfile, 'r') as f:
        for line in f.readlines():
            phrases = line.strip().strip(',').split(',')
            key = int(phrases[0])
            phrases = phrases[1:]
            cluster_map[key] = phrases

    # Get cluster centers
    cluster_centers = {}
    cluster_vecs = {}
    center_file = clusterfile + ".centers"
    calculate_centers = True
    if os.path.exists(center_file):
        cluster_centers = get_cluster_centers(clusterfile)
        calculate_centers = False
    # Get the vectors for each phrase in the clusters
    for num in cluster_map.keys():
        #print "cluster " + str(num)
        vecs = []
        phrases = cluster_map[num]
        for phrase in phrases:
            words = phrase.split(' ')
            word_vecs = []
            for word in words:
                vec = data_util.zero_vec(dim)
                if word in word2vec:
                    vec = word2vec[word]
                word_vecs.append(vec)
            if len(word_vecs) == 0:
                #print "ZERO VEC: " + phrase
                phrase_vec = data_util.zero_vec(dim)
            else:
                phrase_vec = numpy.average(numpy.asarray(word_vecs), axis=0)
            vecs.append(phrase_vec)
        cluster_vecs[num] = vecs
        if calculate_centers:
            cluster_vec = numpy.average(numpy.asarray(vecs), axis=0)
            #print "cluster " + str(num) + " vec shape: " + str(cluster_vec.shape)
            cluster_centers[num] = cluster_vec

    # Get closest phrase
    if return_phrases:
        for num in cluster_map.keys():
            cluster_vec = cluster_centers[num]
            phrases = cluster_map[num]
            vecs = cluster_vecs[num]
            #print 'phrases: ' + str(len(phrases)) + ', vecs: ' + str(len(vecs))
            best_vec = data_util.zero_vec(dim)
            best_phrase = ""
            best_dist = 10000000.0
            for x in range(len(phrases)):
                 phrase = phrases[x]
                 phrase_len = len(phrase.split(' '))
                 phrase_vec = vecs[x]
                 dist_temp = numpy.linalg.norm(phrase_vec-cluster_vec)
                 # Length penalty
                 dist = dist_temp * phrase_len
                 #print "phrase: " + phrase + ", dist: " + str(dist)
                 if dist < best_dist:
                     best_dist = dist
                     best_vec = phrase_vec
                     best_phrase = phrase
            #print "best phrase: " + best_phrase
            cluster_names[num] = best_phrase

    zero_vec = data_util.zero_vec(dim)
    kw_names = []
    for kw_list in labels:
        #print "kw_list: " + str(type(kw_list)) + " : " + str(kw_list)
        if type(kw_list) is str:
            kw_list = kw_list.split(',')
        kw_embeddings = []
        kw_text = []
        for cluster_num in kw_list:
            if cluster_num != '':
                #print "converting cluster " + cluster_num
                num = int(cluster_num)
                vec = cluster_centers[num]
                #print "vec: " + str(len(vec))
                kw_embeddings.append(vec)
                if return_phrases:
                    name = cluster_names[num]
                    kw_text.append(name)
        kw_names.append(kw_text)
        # Pad vectors
        while len(kw_embeddings) < max_length:
            kw_embeddings.insert(0, zero_vec)
        if len(kw_embeddings) > max_length:
            kw_embeddings = kw_embeddings[:max_length]
        #print "kw_embeddings: " + str(len(kw_embeddings))
        cluster_embeddings.append(kw_embeddings)

    if return_phrases:
        # Write cluster name mapping to file
        outname = clusterfile + ".names"
        outfile = open(outname, 'w')
        for key in cluster_names.keys():
            name = cluster_names[key]
            phrases = cluster_map[key]
            outfile.write(str(key) + " : " + name + " : " + str(phrases) + "\n")
        outfile.close()
        return cluster_embeddings, kw_names
    else:
        return cluster_embeddings

'''
    Convert cluster embeddings back to cluster numbers by finding the closest cluster center
    embeddings: a list of lists of vectors (as a python list)
    clusterfile: the name of the cluster file without the .centers extension
    returns: a list of lists of cluster numbers (as a python list)
'''
def embeddings_to_clusters(embeddings, clusterfile):
    cluster_centers = get_cluster_centers(clusterfile)
    cluster_nums = []
    for seq in embeddings:
        cluster_list = []
        for row in seq:
            vec = numpy.asarray(row)
            best_num = 0
            best_dist = float("inf")
            for num in cluster_centers.keys():
                cluster_vec = cluster_centers[num]
                dist = numpy.linalg.norm(vec-cluster_vec)
                if dist < best_dist:
                    best_dist = dist
                    best_num = num
            cluster_list.append(best_num)
        cluster_nums.append(cluster_list)
    return cluster_nums

def get_cluster_centers(filename):
    cluster_centers = {}
    center_file = filename + ".centers"
    #if os.path.isfile(center_file):
    infile = open(center_file, 'r')
    text = infile.read()
    infile.close()
    center_list = eval(text)
    for x in range(len(center_list)):
        cluster_centers[x] = center_list[x]
    #else:
    #    print "WARNING: cluster centers file not found: " + filename
    return cluster_centers
    
if __name__ == "__main__":main()
