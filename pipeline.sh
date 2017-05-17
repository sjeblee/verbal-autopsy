#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

name="svm_symp_train"
fname="symp"
mkdir -p ../../data/$name
python pipeline.py --name $name --model "svm" --modelname $name --train "all" --test "adult" --preprocess "spell,symp" --features "type,symp_train", --featurename $fname | tee ../../data/$name/out-adult-$fname.txt

#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

#python pipeline.py --name "crossval" --train "neonate" --features "type,narr_count" --ex "crossval" | tee ../../data/crossval/out-neonate.txt

echo "Done"
