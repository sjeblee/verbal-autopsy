#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Classify the keywords into categories using word embeddings

from nltk.tokenize import WordPunctTokenizer
from random import shuffle
from sklearn.metrics import classification_report
import argparse
import math
import numpy
import os
import pandas

# Local imports
from pytorch_models import LinearNN, ElmoCNN
import kw_tools

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train', action="store", dest="trainfile")
    argparser.add_argument('--categories', action="store", dest="catfile")
    argparser.add_argument('--kw_map', action="store", dest="kwfile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--vectors', action="store", dest="vecfile")
    argparser.add_argument('--model_type', action="store", dest="model_type")
    argparser.add_argument('--eval', action="store", dest="should_eval")
    argparser.set_default('model_type', 'pubmed')
    argparser.set_default('should_eval', False)
    args = argparser.parse_args()

    if not (args.trainfile and args.vecfile and args.outfile):
        print('usage: ./classify_keywords.py --train [file.csv] --test [file.csv] --out [file.txt] --vectors [file.vectors]')
        exit()

    supervised_classify(trainfile=args.trainfile,
                        cat_file=args.catfile,
                        kw_file=args.kwfile,
                        outfile=args.outfile,
                        vecfile=args.vecfile, model_type=args.model_type, eval=args.should_eval)


''' Classify keywords using a neural network
    trainfile: the file containing a mapping of keywords to the correct keyword category number (kw_map_all.csv)
    cat_file: the file containing a mapping of category numbers to names (categories_fouo.csv)
    kw_file:
    outfile: give the output mapping file a name (kw_map_pred.csv)
    eval: True to split off 10% of the training data to evaluate on, False to train on the whole data
    model: 'pubmed' to use the PubMed word2vec embeddings with a linear nn, 'elmo' to use Elmo embeddings with a CNN
'''
def supervised_classify(trainfile, kw_file, outfile, vecfile, model_type='pubmed', eval=False):
    print('Training keyword classifier')
    num_categories = 43
    #cat_map = kw_tools.load_category_map(cat_file)
    kw_map = kw_tools.load_keyword_map(kw_file)
    keywords, labels, kw_test = keywords_from_csv(trainfile, kw_map, remove_stopwords=True)
    print('keywords:', len(keywords), 'labels:', len(labels), 'kw_test:', len(kw_test))
    labels = [int(x) for x in labels]
    dataset = list(zip(keywords, labels))

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
    if vecfile is not None:
        vec_model, dim = kw_tools.load_w2v(vecfile)
    print('train_keywords:', str(len(train_keywords)), 'train_labels:', str(len(train_labels)))
    print('test_keywords:', str(len(test_keywords)), 'test_labels:', str(len(test_labels)))

    # Tokenize the keyword phrases
    train_X = tokenize_keywords(train_keywords)
    test_X = tokenize_keywords(test_keywords)
    print('train keywords tokenized:', len(train_X))

    # Filter empty keywords
    train_X, train_labels = filter_empty(train_X, train_labels)
    test_X, test_keywords = filter_empty(test_X, test_keywords)

    # Convert labels to arrays
    train_Y = numpy.asarray(train_labels)
    test_Y = numpy.asarray(test_labels)

    print('test_Y[0:10]:', str(test_Y[0:10]))
    print('train_x:', len(train_X), 'train_y:', str(train_Y.shape))
    print('test_x:', len(test_X), 'test_y: ', str(test_Y.shape))
    #num_labels = train_Y.shape[-1]

    # pytorch nn model
    if model_type == 'pubmed':
        train_X = to_embeddings(train_X, vec_model)
        test_X = to_embeddings(test_X, vec_model)
        dim = train_X.shape[-1]
        model = LinearNN(input_size=dim, hidden_size=100, num_classes=num_categories, num_epochs=20)

    # CNN model with Elmo embeddings
    elif model_type == 'elmo':
        dim = 1024 # For ELMo
        model = ElmoCNN(input_size=dim, num_classes=num_categories)

    else:
        print('ERROR: unrecognized model type:', model_type, ' - should be "elmo" or "pubmed"')
        exit(1)

    # Train the model
    model.fit(train_X, train_Y)
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


''' Convert phrases to embeddings (if multiple words, average their embeddings)
    keywords: a list of keywords to convert to embeddings
    vectors: the word embedding model
'''
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


''' Get keywords from csv file
    filename: the csv file
    kw_map: the mapping of keywords to category numbers
    remove_stopwords: True to remove stopwords
'''
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
                cat = int(kw_map[filt_kw]) # Check for the filtered version, but save the original
                if cat == 43:
                    cat = 0
                if kw not in keywords:
                    keywords.append(kw)
                    labels.append(int(cat))
                    in_train = True
            if not in_train:
                kw_test.append(kw)
        #keywords = kw_map.keys()
        #labels = kw_map.values()
    for x in range(len(labels)):
        if labels[x] == 43:
            labels[x] = 0
    return keywords, labels, kw_test


''' Tokenize a list of keywords
    keywords: the list of keywords to tokenize
'''
def tokenize_keywords(keywords):
    kw_tok = []
    tokenizer = WordPunctTokenizer()
    for kw in keywords:
        tok = tokenizer.tokenize(kw)
        #if len(tok) > 0:
        kw_tok.append(tok)
    print('tokenized keywords:')
    for i in range(0, 10):
        print(kw_tok[i])
    return kw_tok


''' Remove empty keyword phrases
    x: the list of keyword phrases
    y: the corresponding list of category labels, so we can keep them matched up
'''
def filter_empty(x, y):
    new_x = []
    new_y = []
    print('filter empty: x:', len(x), 'y:', len(y))
    for i in range(len(x)):
        if len(x[i]) > 0:
            new_x.append(x[i])
            if len(y) > 0:
                new_y.append(y[i])
    return new_x, new_y


if __name__ == "__main__": main()
