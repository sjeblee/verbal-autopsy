#!/bin/bash
num=100
iter=1

# CLUSTER KEYWORDS
./keywords/cluster_keywords.py --in ../../data/crossval_kw/mds+rct/train_adult_${iter}_cat.xml --out ../../data/crossval_kw/mds+rct/train_adult_${iter}_cat_kwkm${num}.xml --clusters ../../data/crossval_kw/mds+rct/train_adult_${iter}_cat.clusters_km${num} --vectors ../../data/datasets/mds+rct/ice+medhelp+narr_all_${iter}.vectors.100 --num ${num} --test ../../data/crossval_kw/mds+rct/test_adult_${iter}_cat.xml --testout ../../data/crossval_kw/mds+rct/test_adult_${iter}_cat_kwkm${num}.xml | tee ../../data/crossval_kw/kw_only/out-cluster-${iter}.txt

# PREDICT KEYWORDS
./keywords/predict_keywords.py --model 'cnn' --train /u/sjeblee/research/va/data/crossval_kw/mds+rct/train_adult_${iter}_cat_kwkm${num}.xml --test /u/sjeblee/research/va/data/crossval_kw/mds+rct/test_adult_${iter}_cat_kwkm${num}.xml --out /u/sjeblee/research/va/data/crossval_kw/multi/test_adult_${iter}_cat_kwkm${num}pred.xml --clusters /u/sjeblee/research/va/data/crossval_kw/mds+rct/train_adult_${iter}_cat.clusters_km${num} --num $num --vectors ../../data/datasets/mds+rct/ice+medhelp+narr_all_${iter}.vectors.100 | tee ../../data/crossval_kw/multi/out-kwpred-${iter}.txt

#./temporal/predict_keywords.py --train ../../data/datasets/mds+rct/train_child_cat_spell_kw.xml --test ../../data/datasets/mds+rct/dev_child_cat_spell_kwnn.xml --out ../../data/datasets/mds+rct/dev_child_cat_spell_kw.xml

name="gru_kwe2"
modelname="gru"
prep="spell,kwc"
feats="narr_vec,kw_clusters"
fname="narrv100_kwc"
labels="cghr_cat"
dataset="mds+rct"
trainset="adult"
devset="adult"
vecfile="/u/sjeblee/research/va/data/datasets/mds+rct/narr+ice+medhelp.vectors.100"
#mkdir -p ../../data/$name

# DEV
#python pipeline_new.py --name $name --model $modelname --dataset $dataset --modelname ${modelname}_${trainset}_${fname}_${labels} --train $trainset --dev $devset --preprocess $prep --features $feats --featurename $fname --labels $labels --vectors $vecfile | tee ../../data/$name/out-$devset-$modelname-$fname.txt
