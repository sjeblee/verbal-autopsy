#!/bin/bash
# Script to run all the steps of the temporal relation classification pipeline

modelname="nn"
relations="contains"
reltype="ee"
trainfile="/u/sjeblee/research/data/thyme/train_dctrel.xml"
testfile="/u/sjeblee/research/data/thyme/dev_dctrel.xml"
vecfile="/u/sjeblee/research/vectors/GoogleNews-vectors-negative300.bin"

# TODO: run time/event tagger

python3 temporal/classify_relations.py --model $modelname --train $trainfile --test $testfile --vectors $vecfile --relset $relations --type $reltype

# TODO: run ranking model

echo "Done"
