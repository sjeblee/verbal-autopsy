#!/usr/bin/python
# -*- coding: utf-8 -*-
# Train word2vec model

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--size', action="store", dest="vec_size")
    args = argparser.parse_args()

    if not (args.infile and args.outfile):
        print "usage: ./word2vec.py --size [vector_size] --in [train_narrs.txt]"
        exit()

    run(args.infile, args.vec_size)

def run(infile, vec_size):

    bin_dir = "~/tools/word2vec/word2vec/bin"

    # Input data
    data_dir = "~/research/va/data/datasets/mds+rct"
    ice_data = "~/research/va/res/ICE-India/Corpus/all-final.txt"
    medhelp_data = "~/research/va/tools/medhelp-crawler/data/posts.txt"

    suffix = "cat_spell.narrsent"

    # Output data
    text_data = data_dir + "/all_narr.txt"
    vec_data = data_dir + "/narr+ice.vectors"

    #cat $DATA_DIR/train_all_$SUFFIX $ICE_DATA $MEDHELP_DATA > $TEXT_DATA

    #rm -f $VECTOR_DATA
    window_size = 5
    num_threads = 12

    if not os.path.exists(vec_data):
        print "--------------------------------------------------------------------"
        print "-- Training vectors..."
        process = subprocess.Popen(["time", bin_dir + "/word2vec", "-train", text_data, "-output", vec_data, "-cbow", 0, "-size", vec_size, "-window", window_size, "-negative", "0", "-hs", "1", "-min-count", "1", "-sample", "1e-3", "-threads", num_threads, "-binary", "0"], stdout=subprocess.PIPE)
        output, err = process.communicate()
        #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 0

        print "-- Training binary vectors..."
        process = subprocess.Popen(["time", bin_dir + "/word2vec", "-train", text_data, "-output", vec_data, "-cbow", 0, "-size", vec_size, "-window", window_size, "-negative", "0", "-hs", "1", "-min-count", "1", "-sample", "1e-3", "-threads", num_threads, "-binary", "1"], stdout=subprocess.PIPE)
        output, err = process.communicate()
        #time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA.bin -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 1
  

    print "-------------------------------------------------------------------------"
    #echo -- distance...
    #$BIN_DIR/distance $VECTOR_DATA.bin

if __name__ == "__main__":main()
