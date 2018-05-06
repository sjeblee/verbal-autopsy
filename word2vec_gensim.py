#!/usr/bin/python
# -*- coding: utf-8 -*-
# Train word2vec model

import argparse
import os
import subprocess
import preprocessing
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.word2vec import LineSentence

import data_util
import extract_features

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--size', action="store", dest="vec_size")
    argparser.add_argument('--stem', action="store_true", dest="stem")
    argparser.add_argument('--out', action="store", dest="outfile")
    args = argparser.parse_args()

    if not (args.infile and args.vec_size):
        print "usage: ./word2vec.py --size [vector_size] --in [train_xxx.xml] --out [outfile.vec] (--stem)"
        exit()

    run(args.infile, args.vec_size, args.outfile, args.stem)

def run(infile, vec_size, outfile, stem=False):

    if not os.path.exists(infile):
        print "Warning: infile does not exist!"

    # Quit if vectors already exist
    if os.path.exists(outfile):
        print "Vectors already exist, quitting"
        return outfile

    sentences = LineSentence(infile)
    #print "sentences: " + str(len(sentences))
    window_size = 5
    num_threads = 12

    print "-- Training vectors..."
    vec_model = Word2Vec(sentences, size=int(vec_size), window=window_size, min_count=2, workers=num_threads, negative=5, sg=0)
    #vec_model = FastText(sentences, size=int(vec_size), window=window_size, min_count=1, word_ngrams=1, min_n=2, max_n=6, workers=num_threads, negative=0)
    vec_data = outfile + ".model"
    vec_model.save(vec_data)
    vec_model.wv.save_word2vec_format(outfile)
    
    print "--------------------------------------------------------------------"
    return outfile

def get(word, model):
    dim = model.vector_size
    if word in model:
        return list(model[word])
    else:
        return data_util.zero_vec(dim)

def load(filename):
    if '.bin' in filename:
        model = load_bin_vectors(filename, True)
    elif 'fasttext' in filename:
        model = FastText.load(filename)
    elif '.wtv' in filename:
        model = Word2Vec.load(filename)
    else:
        model = load_bin_vectors(filename, False)
    dim = model.vector_size
    return model, dim

def load_bin_vectors(filename, bin_vecs=True):
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=bin_vecs, unicode_errors='ignore')
    return word_vectors

if __name__ == "__main__":main()
