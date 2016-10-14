#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

# Parameters
set="child_cat" # all, adult, child, or neonate
labels="ICD_cat" # ICD_cat or Final_code
features="narr_count" # narr_bow, narr_tfidf, kw_bow, kw_tfidf
model="svm" # svm, knn, or nn

# Location of data files
dataloc="/home/sjeblee/Documents/Research/VerbalAutopsy/data/datasets"
resultsloc="/home/sjeblee/Documents/Research/VerbalAutopsy/data/minimal6"

# Setup
mkdir -p $resultsloc
trainset="$dataloc/train_$set.xml"
devset="$dataloc/dev_$set.xml"
trainfeatures="$resultsloc/train_$set.features"
devfeatures="$resultsloc/dev_$set.features"
devresults="$resultsloc/dev_$set.results"

# Feature Extraction
echo "Extracting features..."
python raw2vector.py --in $trainset --out $trainfeatures --labels $labels --features $features
python raw2vector.py --in $devset --out $devfeatures --labels $labels --features $features

# Model
echo "Running svm..."
python svm.py --in $trainfeatures --test $devfeatures --out $devresults --labels $labels --model $model
echo "Done"
