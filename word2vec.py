#!/usr/bin/python
# -*- coding: utf-8 -*-
# Train word2vec model

import argparse
import os
import subprocess
import extract_features
import preprocessing

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--size', action="store", dest="vec_size")
    args = argparser.parse_args()

    if not (args.infile and args.vec_size):
        print "usage: ./word2vec.py --size [vector_size] --in [train_xxx.xml]"
        exit()

    run(args.infile, args.vec_size)

def run(infile, vec_size):

    bin_dir = "/u/sjeblee/tools/word2vec/word2vec/bin"

    # Input data
    data_dir = "/u/sjeblee/research/va/data/datasets/mds+rct"
    ice_data = "/u/sjeblee/research/va/res/ICE-India/Corpus/all-lower-stem.txt"
    medhelp_data = "/u/sjeblee/research/data/medhelp/all_medhelp_clean_stem.txt"

    suffix = ".narrsent"

    # Output data
    text_data = data_dir + "/narr+ice+medhelp_stem.txt"
    vec_data = data_dir + "/narr+ice+medhelp_stem.vectors." + str(vec_size) 

    # Extract narrative text from input file
    narrs = extract_features.get_narrs(infile)
    train_data = infile + suffix
    outfile = open(train_data, "w")
    for narr in narrs:
        # Stemming
        narr = preprocessing.stem(narr)
        outfile.write(narr + "\n")
    outfile.close()

    # Combine all the text
    filenames = [ice_data, medhelp_data, train_data]
    outfile = open(text_data, "w")
    for fname in filenames:
        with open(fname) as inf:
            for line in inf:
                outfile.write(line)
    outfile.close()

    #rm -f $VECTOR_DATA
    window_size = "5"
    num_threads = "12"

    if not os.path.exists(vec_data):
        print "--------------------------------------------------------------------"
        print "-- Training vectors..."
        process = subprocess.Popen(["time", bin_dir + "/word2vec", "-train", text_data, "-output", vec_data, "-cbow", "0", "-size", str(vec_size), "-window", window_size, "-negative", "0", "-hs", "1", "-min-count", "1", "-sample", "1e-3", "-threads", num_threads, "-binary", "0"], stdout=subprocess.PIPE)
        output, err = process.communicate()
        print output
        #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 0

        print "-- Training binary vectors..."
        process = subprocess.Popen(["time", bin_dir + "/word2vec", "-train", text_data, "-output", vec_data + ".bin", "-cbow", "0", "-size", str(vec_size), "-window", window_size, "-negative", "0", "-hs", "1", "-min-count", "1", "-sample", "1e-3", "-threads", num_threads, "-binary", "1"], stdout=subprocess.PIPE)
        output, err = process.communicate()
        print output
        #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA.bin -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 1
  

    print "-------------------------------------------------------------------------"
    #echo -- distance...
    #$BIN_DIR/distance $VECTOR_DATA.bin

if __name__ == "__main__":main()
