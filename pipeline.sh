#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

# Directories for data input and output: SET THESE
prefix="../../data"   # File output prefix
dataloc="../../data/datasets"  # Location of data files (i.e. parent of mds+rct folder

name="va_nn_counts"   # Give the output folder a name, will be created in $prefix
modelname="nn"        # Model type: one of: nn cnn svm rnn gru
prep="spell"          # Preprocessings: any comma-separated list of: none spell stem, kwc
feats="narr_count"    # Features, comma-separated list of: narr_count narr_vec narr_symp dem narr_tfidf lda
fname="narrc"         # Give the featureset a name
labels="cghr_cat"     # ICD categories to use (leave this as cghr_cat)
dataset="mds+rct"     # Dataset to use (leave this as mds+rct)
trainset="adult"      # Training data to use: all adult child neonate
devset="adult"        # Testing data to use: all adult child neonate
vecfile="/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100" # Vectors for narr_vec features

# Create the output directory
mkdir -p $prefix/$name

# DEV
python pipeline.py --name $name --model $modelname --dataset $dataset --modelname ${modelname}_${trainset}_${fname}_${labels} --train $trainset --dev $devset --preprocess $prep --features $feats --featurename $fname --labels $labels --vectors $vecfile --prefix $prefix --dataloc $dataloc | tee ../../data/$name/out-$devset-$modelname-$fname.txt

# HYPEROPT
#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

#trainset="all"
#devset="child"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

#devset="adult"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

# CROSS-VALIDATION
#python pipeline.py --name $name --dataset $dataset --train $trainset --features $feats --featurename $fname --ex "crossval" --model $modelname --preprocess $prep --labels $labels | tee ../../data/${name}/out-$devset-$fname.txt

echo "Done"
