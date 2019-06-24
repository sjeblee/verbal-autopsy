#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Classify the keywords into categories using word embeddings

from random import shuffle
from sklearn.metrics import classification_report
import argparse
import math
import numpy
import os
import pandas

# Local imports
from pytorch_models import LinearNN
import kw_tools

def main():
    #argparser = argparse.ArgumentParser()
    #argparser.add_argument('--train', action="store", dest="trainfile")
    #argparser.add_argument('--test', action="store", dest="testfile")
    #argparser.add_argument('--out', action="store", dest="outfile")
    #argparser.add_argument('--vectors', action="store", dest="vecfile")
    #argparser.set_default('testfile', None)
    #args = argparser.parse_args()

    #if not (args.trainfile and args.vecfile and args.outfile):
    #    print('usage: ./classify_keywords.py --train [file.csv] --test [file.csv] --out [file.txt] --vectors [file.vectors]')
    #    exit()

    #supervised_classify(args.trainfile, args.testfile, args.outfile, args.vecfile)
    #run(args.trainfile, args.testfile, args.outfile, args.vecfile)
    filepath = '/u/sjeblee/research/data/va/child_keywords'
    supervised_classify(trainfile=os.path.join(filepath, 'full_keywords_19Jun2019.csv'),
                        cat_file=os.path.join(filepath, 'categories_fouo.csv'),
                        kw_file=os.path.join(filepath, 'kw_map_all.csv'),
                        outfile=os.path.join(filepath, 'kw_map_pred_june19.csv'),
                        vecfile='/u/sjeblee/research/vectors/wikipedia-pubmed-and-PMC-w2v.bin')


'''
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
'''

def supervised_classify(trainfile, cat_file, kw_file, outfile, vecfile):
    print('Training keyword classifier')
    num_categories = 43
    cat_map = kw_tools.load_category_map(cat_file)
    kw_map = kw_tools.load_keyword_map(kw_file)
    keywords, labels, kw_test = keywords_from_csv(trainfile, kw_map, remove_stopwords=True)
    print('keywords:', len(keywords), 'labels:', len(labels), 'kw_test:', len(kw_test))
    labels = [int(x) for x in labels]
    dataset = zip(keywords, labels)
    #print('dataset:', len(dataset))

    eval = False

    # Split data into train and testfile
    if eval:
        shuffle(dataset)
        num = len(dataset)
        test_num = int(math.floor(0.1 * float(num))) # Use 10% of the data for testing
        train_num = num - test_num
        print('train:', str(train_num), 'test:', str(test_num))
        train_dataset = dataset[0:train_num]
        test_dataset = dataset[train_num:]
        train_keywords, train_labels = zip(*train_dataset)
        test_keywords, test_labels = zip(*test_dataset)
    else:
        train_keywords = keywords
        train_labels = labels
        test_keywords = kw_test
        test_labels = []

    # Encode keywords as word embeddings
    vec_model, dim = kw_tools.load_w2v(vecfile)
    print('train_keywords:', str(len(train_keywords)), 'train_labels:', str(len(train_labels)))
    print('test_keywords:', str(len(test_keywords)), 'test_labels:', str(len(test_labels)))

    print('train_keywords:', str(train_keywords))
    train_X = to_embeddings(train_keywords, vec_model)
    test_X = to_embeddings(test_keywords, vec_model)
    train_Y = numpy.asarray(train_labels)
    test_Y = numpy.asarray(test_labels)
    print('train_x: ', str(train_X.shape), ' train_y: ', str(train_Y.shape))
    print('test_x: ' + str(test_X.shape) + ' test_y: ' + str(test_Y.shape))
    dim = train_X.shape[-1]
    #num_labels = train_Y.shape[-1]

    # pytorch nn model
    model = LinearNN(input_size=dim, hidden_size=100, num_classes=num_categories)
    model.fit(train_X, train_Y, 100, 'relu', num_epochs=20)
    pred_y = model.predict(test_X)
    print('pred_y:', str(len(pred_y)), pred_y)

    # Print metrics
    if eval:
        print(classification_report(test_Y, pred_y))
    else:
        # Write the predicted keyword map to csv file
        df = pandas.DataFrame(columns=['terms', 'category'])
        assert(len(pred_y) == len(test_keywords))

        # Add the original map
        print('saving original map...')
        for key in kw_map.keys():
            df = df.append({'terms': key, 'category': kw_map[key]}, ignore_index=True)

        # Add the predicted mappings
        print('saving pred keywords')
        for x in range(len(test_keywords)):
            kw = test_keywords[x].strip()
            category = pred_y[x]

            if len(kw) > 0:
                df = df.append({'terms': kw, 'category': category}, ignore_index=True)
                if x % 1000 == 0:
                    print(str(x), kw, str(category))

        df.to_csv(outfile)


def to_embeddings(keywords, vectors):
    x = []
    for kw in keywords:
        vecs = []
        for word in kw.split(' '):
            vec = kw_tools.get_w2v(word, vectors)
            vecs.append(vec)
        x.append(numpy.average(vecs, axis=0))
    return numpy.asarray(x)


def map_back(clusters, cluster_names):
    cluster_vals = []
    for c_vec in clusters:
        val = numpy.argmax(c_vec)
        cluster_vals.append(cluster_names[val])
    return cluster_vals


def keywords_from_csv(filename, kw_map, remove_stopwords=False):
    df = pandas.read_csv(filename)
    stopwords = []
    with open("stopwords_small.txt", "r") as f:
        for line in f.readlines():
            stopwords.append(line.strip())
    #keywords = []
    kw_test = []
    #labels = []
    keywords = list(kw_map.keys())
    labels = list(kw_map.values())

    filtered_kw_map = {}
    if remove_stopwords:
        for key in kw_map:
            val = kw_map[key]
            filtered_words = [word for word in key.split(' ') if word not in stopwords]
            new_key = ' '.join(filtered_words).strip()
            if len(new_key) > 0:
                filtered_kw_map[new_key] = val
        print('Removing stopwords from kw_map. Original:', len(kw_map), 'Filtered:', len(filtered_kw_map))
        kw_map = filtered_kw_map

    for i, row in df.iterrows():
        kw1 = str(df.at[i, 'p1_keywords'])
        kw2 = str(df.at[i, 'p2_keywords'])
        #print('kw1:', kw1, 'kw2:', kw2)
        row_keywords = kw1 + kw2
        row_keywords = kw_tools.clean_keywords(row_keywords)
        kws = row_keywords.split(',')
        #kws = str(row['keywords_fixed']).split(',')
        for kw in kws:
            kw = kw.strip()
            in_train = False

            # Remove stopwords to check for match
            if remove_stopwords:
                filtered_words = [word for word in kw.split(' ') if word not in stopwords]
                filt_kw = ' '.join(filtered_words)
            else:
                filt_kw = kw

            if len(filt_kw) > 0 and filt_kw in kw_map:
                cat = kw_map[filt_kw] # Check for the filtered version, but save the original
                if kw not in keywords:
                    keywords.append(kw)
                    labels.append(int(cat))
                    in_train = True
            if not in_train:
                kw_test.append(kw)
        #keywords = kw_map.keys()
        #labels = kw_map.values()
    return keywords, labels, kw_test


if __name__ == "__main__": main()
