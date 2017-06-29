#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

name="phmrc_mdstrain"
fname="narrc"
dataset="phmrc+mds_one"
trainset="all"
devset="adult"
model="nn"
mkdir -p ../../data/$name

python pipeline-phmrc.py --name $name --model $model --modelname ${name}_${trainset}_${fname} --dataset $dataset --train $trainset --dev $devset --preprocess "none" --features "type,narr_count", --featurename $fname | tee ../../data/$name/out-$devset-$model-dev.txt

#devset="child"
#python pipeline-phmrc.py --name $name --model $model --modelname ${name}_${trainset}_${fname} --dataset $dataset --train $trainset --test $devset --preprocess "none" --features "type,narr_count", --featurename $fname | tee ../../data/$name/out-$devset-$model-test.txt

#devset="neonate"
#trainset="child+neo"
#python pipeline-phmrc.py --name $name --model $model --modelname ${name}_${trainset}_${fname} --dataset $dataset --train $trainset --test $devset --preprocess "none" --features "type,narr_count", --featurename $fname | tee ../../data/$name/out-$devset-$model-test.txt

#python pipeline-phmrc.py --name $name --model $model --modelname "${model}_h" --dataset $dataset --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --ex "hyperopt" --preprocess "none" | tee ../../data/${name}/out-${devset}-${model}-hyperopt.txt

#python pipeline-phmrc.py --name $name --dataset $dataset --train $trainset --features "type,narr_count" --featurename $fname --ex "crossval" --preprocess "none" | tee ../../data/${name}/out-$trainset.txt

echo "Done"
