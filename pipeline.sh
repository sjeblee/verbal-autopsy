#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

# Parameters
trainname="all_cat" # all, adult, child, or neonate
devname="neonate_cat"
pre="spell" # spell, heidel
labels="ICD_cat" # ICD_cat or Final_code
featureset="kw" # Name of the feature set for feature file
features="type,kw_count" # type, checklist, narr_bow, narr_tfidf, narr_count, kw_bow, kw_tfidf
model="nn" # svm, knn, nn, lstm
modelname="nn1a-all"

# Location of data files
dataloc="/u/sjeblee/research/va/data/datasets"
resultsloc="/u/sjeblee/research/va/data/nn1_kw"
heideldir="/u/sjeblee/tools/heideltime/heideltime-standalone"
scriptdir=$(pwd)

# Setup
mkdir -p $resultsloc
trainset="$dataloc/train_$trainname.xml"
devset="$dataloc/dev_$devname.xml"
trainfeatures="$resultsloc/train_$trainname.features.$featureset"
devfeatures="$resultsloc/dev_$devname.features.$featureset"
devresults="$resultsloc/dev_$devname.results.$modelname.$featureset"

# Preprocessing
spname="sp"
echo "Preprocessing..."
if [[ $pre == *"spell"* ]]
then
    echo "Running spelling correction..."
    trainsp="$dataloc/train_${trainname}_${spname}.xml"
    devsp="$dataloc/dev_${devname}_${spname}.xml"
    if [ ! -f $trainsp ]; then
	python spellcorrect.py --in $trainset --out $trainsp
    fi
    if [ ! -f $devsp ]; then
	python spellcorrect.py --in $devset --out $devsp
    fi

   trainset=$trainsp
   devset=$devsp
fi

if [[ $pre == *"heidel"* ]]
then
    echo "Running Heideltime..."
    cd $heideldir
    trainh="$dataloc/train_${trainname}_ht.xml"
    if [ ! -f $trainh ]; then
	python $scriptdir/heidel_tag.py --in $trainset --out $trainh
	sed -e 's/&lt;/</g' $trainh
	sed -e 's/&gt;/>/g' $trainh
    fi
    trainset=$trainh
    devh="$dataloc/dev_${devname}_ht.xml"
    if [ ! -f $devh ]; then
	python $scriptdir/heidel_tag.py --in $devset --out $devh
	sed -e 's/&lt;/</g' $devh
	sed -e 's/&gt;/>/g' $devh
    fi
    devset=$devh
    cd $scriptdir
fi

# Feature Extraction
if [ ! -f $trainfeatures ] || [ ! -f $devfeatures ]
then
    echo "Extracting features..."
    python extract_features.py --train $trainset --trainfeats $trainfeatures --test $devset --testfeats $devfeatures --labels $labels --features $features
fi

# Model
#if [[ "$model" -eq "svm" ]]
#then
#    echo "Running svm..."
#    python svm.py --in $trainfeatures --test $devfeatures --out $devresults --labels $labels --model $model
if [[ "$model" -eq "nn" || "$model" -eq "lstm" ]]
then
    echo "Running nn..."
    python nn.py --in $trainfeatures --test $devfeatures --out $devresults --labels $labels --model $model --prefix $resultsloc --name $modelname
fi

# Results statistics
echo "Calculating scores..."
python results_stats.py --in $devresults --out $devresults.stats

echo "Done"
