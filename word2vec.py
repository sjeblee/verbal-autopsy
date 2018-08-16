#!/usr/bin/python
# -*- coding: utf-8 -*-
# Train word2vec model

import argparse
import os
import subprocess
import preprocessing
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.fasttext import FastText

import data_util
import extract_features

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--size', action="store", dest="vec_size")
    argparser.add_argument('--stem', action="store_true", dest="stem")
    argparser.add_argument('--name', action="store", dest="name")
    args = argparser.parse_args()

    if not (args.infile and args.vec_size):
        print "usage: ./word2vec.py --size [vector_size] --in [train_xxx.xml] --name [name] (--stem)"
        exit()

    run(args.infile, args.vec_size, args.name, args.stem)

def run(infile, vec_size, name="narr+ice+medhelp", stem=False):

    #bin_dir = "/u/sjeblee/tools/word2vec/word2vec/bin"
    bin_dir = "/u/yoona/word2vec/bin"

    # Input data
    #data_dir = "/u/sjeblee/research/va/data/datasets/mds+rct"
    data_dir = "/u/yoona/test/mds+rct" # hard-coded, Yoona's data location
    #ice_data = "/u/sjeblee/research/data/ICE-India/Corpus/all-lower.txt"
    ice_data = "/u/yoona/mds+rct/ice_all_lower.txt"
    #medhelp_data = "/u/sjeblee/research/data/medhelp/all_medhelp_clean_lower.txt"
    medhelp_data = "/u/yoona/mds+rct/all_medhelp_clean_lower.txt"
    suffix = ".narrsent"
    if stem:
        ice_data = "/u/sjeblee/research/data/ICE-India/Corpus/all-lower-stem.txt"
        medhelp_data = "/u/sjeblee/research/data/medhelp/all_medhelp_clean_stem.txt"
        suffix = ".narrsent.stem"

    # Output data
    text_data = data_dir + "/" + name + ".txt"
    vec_data = data_dir + "/" + name + ".vectors." + str(vec_size)
    if stem:
        text_data = data_dir + "/" + name + "_stem.txt"
        vec_data = data_dir + "/" + name + "_stem.vectors." + str(vec_size)

    # Quit if vectors already exist
    if os.path.exists(vec_data):
        print "Vectors already exist, quitting"
        return vec_data

    # Extract narrative text from input file
    narrs = extract_features.get_narrs(infile)
    train_data = infile + suffix
    outfile = open(train_data, "w")
    for narr in narrs:
        if stem:
            narr = preprocessing.stem(narr)
        outfile.write(narr + "\n")
    outfile.close()

    # Combine all the text
    #filenames = [ice_data, medhelp_data, train_data]
    filenames = [train_data]
    #filenames = [medhelp_data, train_data]
    sentences = []
    outfile = open(text_data, "w")
    for fname in filenames:
        with open(fname) as inf:
            for line in inf.readlines():
                line = unicode(line, errors='ignore')
                outfile.write(line.strip().encode('utf8'))
                sentences.append(line.strip())
    outfile.close()

    #rm -f $VECTOR_DATA
    window_size = 5
    num_threads = 12
    #sentences = []
    #with open(outfile) as f:
    #    for line in f.readlines():
    #        sentences.append(line.strip())

    print "-- Training vectors..."
    #vec_model = Word2Vec(sentences, size=int(vec_size), window=window_size, min_count=1, workers=num_threads, negative=0, sg=1)
    #vec_model = FastText(sentences, size=int(vec_size), window=window_size, min_count=1, word_ngrams=1, min_n=2, max_n=6, workers=num_threads, negative=0)
    #vec_model.save(vec_data)
       
    if not os.path.exists(vec_data):
        print "--------------------------------------------------------------------"
        process = subprocess.Popen(["time", bin_dir + "/word2vec", "-train", text_data, "-output", vec_data, "-cbow", "1", "-size", str(vec_size), "-window", str(window_size), "-negative", "0", "-hs", "1", "-min-count", "1", "-sample", "1e-3", "-threads", str(num_threads), "-binary", "0"], stdout=subprocess.PIPE)
        output, err = process.communicate()
        print output
        #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 0

        print "-- Training binary vectors..."
        process = subprocess.Popen(["time", bin_dir + "/word2vec", "-train", text_data, "-output", vec_data + ".bin", "-cbow", "1", "-size", str(vec_size), "-window", str(window_size), "-negative", "0", "-hs", "1", "-min-count", "1", "-sample", "1e-3", "-threads", str(num_threads), "-binary", "1"], stdout=subprocess.PIPE)
        output, err = process.communicate()
        print output
        #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA.bin -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 1

    print "-------------------------------------------------------------------------"
    #echo -- distance...
    #$BIN_DIR/distance $VECTOR_DATA.bin
   
    return vec_data

def get(word, model):
    dim = model.vector_size
    if word in model: #.wv.vocab:
        return list(model[word])
    else:
        return data_util.zero_vec(dim)
    #return model[word]
    #return model.get_vector(word)

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
