#!/bin/bash
# Usage: ./addcategories.sh [dataset] [name] i.e. adult train

dataset=$1
name=$2
folder="agincourt"
prefix="/u/sjeblee/research/va/data"

echo $dataset $name

#./addcategory.py --in $prefix/datasets/${folder}/${name}_${dataset}_cat.xml --out $prefix/datasets/${folder}/${name}_${dataset}_cat_0.xml --map $prefix/icd_map.csv --label "ICD_cat"

./addcategory.py --in $prefix/datasets/${folder}/${name}_${dataset}_cat.xml --out $prefix/datasets/${folder}/${name}_${dataset}_cat_1.xml --map $prefix/icd_map_ccodex4.csv --label "WB10_codex4"

./addcategory.py --in $prefix/datasets/${folder}/${name}_${dataset}_cat_1.xml --out $prefix/datasets/${folder}/${name}_${dataset}_cat_2.xml --map $prefix/icd_map_ccodex2.csv --label "WB10_codex2"

./addcategory.py --in $prefix/datasets/${folder}/${name}_${dataset}_cat_2.xml --out $prefix/datasets/${folder}/${name}_${dataset}_cat_3.xml --map $prefix/icd_map_ccodex.csv --label "WB10_codex"

./addcategory.py --in $prefix/datasets/${folder}/${name}_${dataset}_cat_3.xml --out $prefix/datasets/${folder}/${name}_${dataset}_cat.xml --map $prefix/cghr_who_code_map_${dataset}.csv --label "cghr_cat"

rm $prefix/${folder}/${name}_${dataset}_cat_1.xml
rm $prefix/${folder}/${name}_${dataset}_cat_2.xml
rm $prefix/${folder}/${name}_${dataset}_cat_3.xml
