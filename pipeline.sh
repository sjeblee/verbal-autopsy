#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

name="baseline_lstm_medttk"
modelname="lstm"
fname="narr_vec"
dataset="mds_one+tr"
trainset="all"
devset="adult"
mkdir -p ../../data/$name

python pipeline.py --name $name --model $modelname --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --preprocess "spell,medttk" --features "narr_vec", --featurename $fname | tee ../../data/$name/out-$devset-$modelname-$fname.txt

#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

#trainset="all"
#devset="child"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

#devset="adult"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt


#python pipeline.py --name $name --dataset $dataset --train $trainset --features "type,narr_count" --featurename $fname --ex "crossval" --preprocess "spell" | tee ../../data/${name}/out-$devset-$fname.txt

echo "Done"
