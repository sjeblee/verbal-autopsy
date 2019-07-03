#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Cluster the keywords from the records using word2vec

from collections import Counter
from lxml import etree
from numpy import array
from sklearn.cluster import AgglomerativeClustering, MiniBatchKMeans, SpectralClustering, KMeans
from sklearn.metrics import calinski_harabaz_score
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats import mode
import argparse
import numpy
import os
import time
import sys

import kw_tools

numpy.set_printoptions(threshold=sys.maxsize)

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--clusters', action="store", dest="clusterfile")
    argparser.add_argument('--vectors', action="store", dest="vecfile")
    #argparser.add_argument('--test', action="store", dest="testfile")
    #argparser.add_argument('--testout', action="store", dest="testoutfile")
    argparser.add_argument('-n', '--num', action="store", dest="num") # number of clusters
    argparser.set_default('num', 100)
    args = argparser.parse_args()

    if not (args.infile and args.clusterfile and args.vecfile):
        print('usage: ./cluster_keywords.py --in [file.csv] --out [file.csv] --vecfile [file.bin] --num [n_clusters]')
        exit()

    unsupervised_cluster(args.infile, args.vecfile, args.clusterfile, args.outfile, args.num)


''' Do unsupervised clustering of keywords
    infile: the csv file with keywords to use
    vecfile: the word embedding file
    clusterfile: give a name for the output file that will store the clusters
    outfile: give a name for the output mapping of keyword to cluster number (need this file for extracting features later)
    num_clusters: how many clusters to generate
'''
def unsupervised_cluster(infile, vecfile, clusterfile, outfile, num_clusters=100):
    # Load word embeddings
    start = time.time()
    print('loading word embedddings...')
    vec_model, dim = kw_tools.load_w2v(vecfile)
    print('loading embeddings took', (time.time() - start), 's')

    # Load keyword phrases from csv
    csv_keywords = kw_tools.get_keywords_from_csv(infile)

    # Remove blank keywords and duplicates
    keywords = []
    for kw in csv_keywords:
        kw = kw.strip() # Remove extra whitespace
        if len(kw) > 0:
            #if kw not in keywords: # Uncomment to remove duplicates
            keywords.append(kw)

    # Extract keywords
    kw_vecs = []
    num_clusters = int(num_clusters)
    print('num_keywords:', str(len(keywords)))
    print('num_clusters:', str(num_clusters))
    print('dim:', str(dim))

    # Get the embeddings for each keyword
    for phrase in keywords:
        vec = vectorize(phrase, vec_model, dim)
        kw_vecs.append(vec)

    # Generate clusters
    kw_vecs_np = numpy.asarray(kw_vecs)
    print('shape: [num_keywords, dim]:', str(kw_vecs_np.shape)) # keywords listed individually
    print('kw_vecs_np[0]:', str(kw_vecs_np[0]))
    print('generating clusters...')
    #clusterer = MiniBatchKMeans(n_clusters=num_clusters, max_iter=1000, n_init=1000, batch_size=500, reassignment_ratio=0.03, max_no_improvement=50)
    #clusterer = MiniBatchKMeans(n_clusters=num_clusters)
    clusterer = KMeans(n_clusters=num_clusters, max_iter=500)
    #clusterer = SpectralClustering(n_clusters=num_clusters, n_init=15, affinity='cosine', n_jobs=-1)
    #clusterer = AgglomerativeClustering(n_clusters=num_clusters, affinity='cosine', linkage='average')
    kw_clusters = clusterer.fit_predict(kw_vecs_np)
    #kw_clusters = map_back(clusterer.fit_predict(kw_vecs), cluster_names)

    # Unsupervised cluster metrics
    cluster_labels = clusterer.labels_
    cluster_centers = list(clusterer.cluster_centers_)
    chi_score = calinski_harabaz_score(kw_vecs, cluster_labels)
    print('Calinski-Harabaz Index:', str(chi_score))

    print('Writing clusters to csv file...')
    write_clusters_to_file(clusterfile, get_cluster_map(keywords, kw_clusters))

    # Save the cluster centers
    outf = open(clusterfile + ".centers", 'w')
    outf.write(str(cluster_centers))
    outf.close()

    # Get text interpretation of clusters
    interpret_clusters(clusters=None, clusterfile=clusterfile)

    # Write results to file
    outf = open(outfile, 'w')
    outf.write('terms,category\n')
    for x in range(len(keywords)):
        outf.write(keywords[x] + ',' + str(kw_clusters[x]) + '\n')
    outf.close()
    print('Total time:', ((time.time() - start)/60), 'mins')


def read_cluster_file(clusterfile, word2vec, dim, cluster_names=None):
    train = False
    if cluster_names is None:
        cluster_names = set()
        cluster_names.add(0)
        train = True
    keywords = []
    kw_vecs = []
    kw_clusters = []

    zero_vec = numpy.array(kw_tools.zero_vec(dim))

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
        print('correct_labels:', str(correct_labels))
        m, count = mode(numpy.array(correct_labels))
        label = m[0]
        print('label:', str(label))
        for item in correct_labels:
            n = n+1
            if item == label:
                n_corr = n_corr+1

    # Calculate purity score
    print('n:', str(n))
    print('n_corr:', str(n_corr))
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


''' Write the generated clusters to a file
    outfile: the file to write the clusters to
    cluster_map: a python dictionary of cluster number to a list of phrases
'''
def write_clusters_to_file(outfile, cluster_map):
    print('writing clusters to', outfile)
    outf = open(outfile, 'w')
    for key in cluster_map:
        print('cluster', str(key), ':', str(len(cluster_map[key])), 'keywords')
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
def write_clusters_to_xml(xmlfile, outfile, ids, cluster_pred, text_pred=None, kw_label='keyword_clusters'):
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
        #if narr_node is not None:
        #    narr = narr_node.text
        #    kw_textrank = textrank.extract_key_phrases(narr)
            #print "kw_textrank: " + str(kw_textrank)
        #    tr_node = etree.SubElement(child, 'textrank_keyphrases')
        #    tr_node.text = str(kw_textrank)
        keyword_text = ''
        # Edit by Yoona. For keywords vector concatenation
        keyword_labels = ''
        if rec_id in id_dict:
            keywords = id_dict[rec_id]
            kw_text = text_dict[rec_id]
            for kw in keywords:
                keyword_text = keyword_text + "," + str(kw)
            # Edit by Yoona for Keyword concatenation
            for kwlabels in kw_text:
                keyword_labels = keyword_labels + " " + str(kwlabels)
            newnode = etree.SubElement(child, kw_label)
            newnode.text = keyword_text.strip(',')
            newnode2 = etree.SubElement(child, kw_label + '_text')
            #newnode2.text = str(kw_text)
            newnode2.text = str(keyword_labels)
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


''' Convert the keyword phrases to embeddings for clustering
    phrase: the keyword phrase
    vec_model: the embedding model
    dim: the number of dimensions of the word embeddings
'''
def vectorize(phrase, vec_model, dim):
    stopwords = ['and', 'or', 'with', 'in', 'of', 'at', 'had', 'ho']
    words = phrase.split(' ')
    vecs = []
    zero_vec = numpy.asarray(kw_tools.zero_vec(dim))
    for word in words:
        if word not in stopwords:
            vecs.append(kw_tools.get_w2v(word, vec_model))
    # Average vectors
    if len(vecs) > 0:
        avg_vec = numpy.average(numpy.asarray(vecs), axis=0)
        return avg_vec
    else:
        return zero_vec


def clusters_from_file(clusterfile):
    print('TODO')


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
    outname = clusterfile + '.names'
    outfile = open(outname, 'w')
    outfile.write('No. ,Final categories\n')
    key_list = sorted(list(cluster_names.keys()))
    for key in key_list:
        name = cluster_names[key]
        phrases = cluster_map[key]
        outfile.write(str(key) + ',' + name + '\n') #+ ' : ' + str(phrases) + '\n')
    outfile.close()

    # Replace cluster names with text
    if clusters is not None:
        for entry in clusters:
            text = []
            for num in entry.split(','):
                if num != '':
                    if int(num) in cluster_names:
                        name = cluster_names[int(num)]
                        text.append(name)
                    else:
                        print('NOT FOUND:', num)
            clusters_text.append(text)
            return clusters_text
    else:
        return None


''' Get cluster embeddings for keyword clusters
    labels: list of lists of cluster numbers
'''
def cluster_embeddings(labels, clusterfile, vecfile, return_phrases=False, max_length=10):
    print('getting cluster embeddings')
    word2vec, dim = kw_tools.load_w2v(vecfile)
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
    center_file = clusterfile + '.centers'
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
                vec = kw_tools.zero_vec(dim)
                if word in word2vec:
                    vec = word2vec[word]
                word_vecs.append(vec)
            if len(word_vecs) == 0:
                #print "ZERO VEC: " + phrase
                phrase_vec = kw_tools.zero_vec(dim)
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
            best_vec = kw_tools.zero_vec(dim)
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

    #zero_vec = kw_tools.zero_vec(dim)
    kw_names = []

    if return_phrases:
        # Write cluster name mapping to file
        outname = clusterfile + '.names'
        outfile = open(outname, 'w')
        outfile.write('No. ,Final categories\n')
        key_list = sorted(list(cluster_names.keys()))
        for key in key_list:
            name = cluster_names[key]
            phrases = cluster_map[key]
            outfile.write(str(key) + ',' + name + '\n') #+ ' : ' + str(phrases) + '\n')
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
            best_dist = float('inf')
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
    center_file = filename + '.centers'
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


if __name__ == "__main__": main()
