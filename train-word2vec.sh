#!/bin/bash
# Train word2vec model

if [ $# -ne 1 ]; then
    echo $0: usage: .train-word2vec.sh [vector_size] [train_narrs]
    exit 1
fi

BIN_DIR=$HOME/tools/word2vec/word2vec/bin
DIM=$1
NARR_DATA=$2

# Input data
DATA_DIR=$HOME/research/va/data/datasets/mds+rct
ICE_DATA=$HOME/research/va/data/ice # TODO
MEDHELP_DATA=$HOME/research/va/tools/medhelp-crawler/data/posts.txt

#SUFFIX="cat_spell.narrsent"

# Output data
TEXT_DATA=$DATA_DIR/all_narr+medhelp+ice.txt
VECTOR_DATA=$DATA_DIR/narr+medhelp+ice.vectors.$DIM

cat $NARR_DATA $ICE_DATA $MEDHELP_DATA > $TEXT_DATA

#rm -f $VECTOR_DATA

if [ ! -e $VECTOR_DATA ]; then
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 0

  echo "-- Training binary vectors..."
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA.bin -cbow 0 -size $DIM -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 1
  
fi

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $VECTOR_DATA.bin
