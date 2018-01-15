#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

name="dirichlet_neo"
modelname="nn"
prep="spell"
feats="type,narr_count"
fname="narrc"
labels="ICD_cat_neo"
dataset="mds+rct_small"
trainset="neonate"
devset="neonate"
mkdir -p ../../data/$name

# DEV
#python pipeline.py --name $name --model $modelname --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --preprocess $prep --features $feats --featurename $fname --labels $labels | tee ../../data/$name/out-$devset-$modelname-$fname.txt

# HYPEROPT
#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

#trainset="all"
#devset="child"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

#devset="adult"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

# CROSSVAL
python pipeline_dirichlet.py --name $name --dataset $dataset --train $trainset --features $feats --featurename $fname --ex "crossval" --model $modelname --preprocess $prep --labels $labels | tee ../../data/${name}/out-$devset-$fname.txt

echo "Done"
