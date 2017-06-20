#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

name="crossval_bmc"
fname="narrc"
trainset="child"
devset="adult"
mkdir -p ../../data/$name

#python pipeline.py --name $name --model "nn" --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --preprocess "spell,lemma" --features "type,narr_count", --featurename $fname | tee ../../data/$name/out-$devset-$fname.txt

#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

python pipeline.py --name $name --train $trainset --features "type,narr_count" --featurename $fname --ex "crossval" --preprocess "spell,stem" | tee ../../data/${name}/out-$trainset.txt

echo "Done"
