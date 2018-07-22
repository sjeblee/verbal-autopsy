#a!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

# Directories for data input and output: SET THESE
prefix="/u/yoona/data/test/arg_symp_cv"   # File output prefix
dataloc="/u/yoona/test"  # Location of data files (i.e. parent of mds+rct folder

name="va_cnn_vec_symp"   # Give the output folder a name, will be created in $prefix
modelname="cnn"        # Model type: one of: nn cnn svm rnn gru
prep="spell,symp"          # Preprocessings: any comma-separated list of: none spell stem, kwc
feats="narr_vec"    # Features, comma-separated list of: narr_count narr_vec narr_symp dem narr_tfidf lda
fname="narrv"         # Give the featureset a name
labels="cghr_cat"     # ICD categories to use (leave this as cghr_cat)
dataset="mds+rct"     # Dataset to use (leave this as mds+rct) or agincourt
trainset="adult"      # Training data to use: all adult child neonate
devset="adult"        # Testing data to use: all adult child neonate
vecfile="/u/yoona/mds+rct/narr+ice+medhelp.vectors.100" # Vectors for narr_vec features
sympfile="/u/yoona/symptom_files/SYMP.csv" # Symptom file for symptom extraction
chvfile="/u/yoona/symptom_files/CHV_concepts_terms_flatfile_20110204.tsv" # CHV file for symptom extraction

# Create the output directory
mkdir -p $prefix/$name

# DEV
#python pipeline.py --name $name --model $modelname --dataset $dataset --modelname ${modelname}_${trainset}_${fname}_${labels} --train $trainset --dev $devset --preprocess $prep --features $feats --featurename $fname --labels $labels --vectors $vecfile --prefix $prefix --dataloc $dataloc --sympfile $sympfile --chvfile $chvfile | tee $prefix/$name/out-$devset-$modelname-$fname.txt

# HYPEROPT
#python pipeline.py --name "rf_hyperopt" --model "rf" --modelname "rf_h" --train "all" --dev "all" --features "type,narr_count" --ex "hyperopt" | tee ../../data/rf_hyperopt/out-hyperopt.txt

#trainset="all"
#devset="child"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

#devset="adult"
#python pipeline.py --name $name --model "nn" --dataset $dataset --modelname ${name}_${trainset}_${fname} --train $trainset --dev $devset --features "type,narr_count" --featurename $fname --preprocess "spell,medttk" | tee ../../data/${name}/out-$devset-$fname-nn.txt

# CROSS-VALIDATION
python pipeline.py --name $name --dataset $dataset --train $trainset --features $feats --featurename $fname --ex "crossval" --model $modelname --preprocess $prep --labels $labels --dataloc $dataloc --prefix $prefix --vectors $vecfile --sympfile $sympfile --chvfile $chvfile | tee ../../data/${name}/out-$devset-$fname.txt

echo "Done"
