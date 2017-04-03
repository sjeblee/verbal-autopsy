#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

python pipeline.py --name "svm_tfidf2" --model "svm" --modelname "svm_tfidf2" --train "all" --test "adult" --features "type,narr_tfidf", --featurename "tfidf" | tee ../../data/svm_tfidf2/out-adult-tfidf.txt

#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

#python pipeline.py --name "crossval" --train "neonate" --features "type,narr_count" --ex "crossval" | tee ../../data/crossval/out-neonate.txt

echo "Done"
