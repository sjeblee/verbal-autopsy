#!/bin/bash
# Script to run all the steps of the temporal relation classification pipeline

modelname="nn"
relations="exact"
reltype="ee"
trainfile="/u/sjeblee/research/data/thyme/train.xml"
testfile="/u/sjeblee/research/data/thyme/dev.xml"
vecfile="/u/sjeblee/research/vectors/GoogleNews-vectors-negative300.bin"

python3 temporal/classify_relations.py --model $modelname --train $trainfile --test $testfile --vectors $vecfile --relset $relations --type $reltype

echo "Done"
