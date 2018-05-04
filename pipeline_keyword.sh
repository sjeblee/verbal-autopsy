#!/bin/bash

# CLUSTER KEYWORDS
./keywords/cluster_keywords.py --in ../../data/datasets/mds+rct/train_adult_cat_spell.xml --out ../../data/datasets/mds+rct/train_adult_cat_spell_kwkm100.xml --clusters ../../data/datasets/mds+rct/train_adult_cat_spell.clusters_km100 --vectors ../../data/datasets/mds+rct/narr+ice+medhelp.vectors.100 --num 100 --test "../../data/datasets/mds+rct/dev_adult_cat_spell.xml" --testout "../../data/datasets/mds+rct/dev_adult_cat_spell_kwkm100nn.xml"

# PREDICT KEYWORDS
./keywords/predict_keywords.py --model 'encoder-decoder' --train ../../data/datasets/mds+rct/train_adult_cat_spell_kwkm100.xml --test ../../data/datasets/mds+rct/dev_adult_cat_spell_kwkm100nn.xml --out ../../data/datasets/mds+rct/dev_adult_cat_spell_kwkm100pred.xml --clusters ../../data/datasets/mds+rct/train_adult_cat_spell.clusters_km100 --num 100

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
