#!/bin/bash
# Train word2vec model

DATA_DIR=$HOME/research/va/data/datasets/mds_one
BIN_DIR=$HOME/tools/word2vec/word2vec/bin

SUFFIX="cat_spell.narrsent"

TEXT_DATA=$DATA_DIR/all_narr+ice.txt
VECTOR_DATA=$DATA_DIR/narr+ice.vectors

#cat $DATA_DIR/train_all_$SUFFIX $DATA_DIR/dev_all_$SUFFIX $DATA_DIR/test_all_$SUFFIX > $TEXT_DATA

#rm -f $VECTOR_DATA

if [ ! -e $VECTOR_DATA ]; then
  echo -----------------------------------------------------------------------------------------------------
  echo -- Training vectors...
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 0

  echo "-- Training binary vectors..."
  time $BIN_DIR/word2vec -train $TEXT_DATA -output $VECTOR_DATA.bin -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -min-count 1 -sample 1e-3 -threads 12 -binary 1
  
fi

echo -----------------------------------------------------------------------------------------------------
echo -- distance...

$BIN_DIR/distance $VECTOR_DATA.bin
