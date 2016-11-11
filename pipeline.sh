#!/bin/bash
# Script to run all the steps of the Verbal Autopsy pipeline

# Parameters
trainname="adult_cat" # all, adult, child, or neonate
devname="adult_cat"
pre="spell" # spell, heidel
labels="ICD_cat" # ICD_cat or Final_code
featureset="narr_tfidf" # Name of the feature set for feature file
features="type,narr_tfidf" # type, checklist, narr_bow, narr_tfidf, kw_bow, kw_tfidf
model="svm" # svm, knn, or nn

# Location of data files
dataloc="/u/sjeblee/research/va/data/datasets"
resultsloc="/u/sjeblee/research/va/data/svm_tfidf"
heideldir="/u/sjeblee/tools/heideltime/heideltime-standalone"
scriptdir=$(pwd)

# Setup
mkdir -p $resultsloc
trainset="$dataloc/train_$trainname.xml"
devset="$dataloc/dev_$devname.xml"
trainfeatures="$resultsloc/train_$trainname.features.$featureset"
devfeatures="$resultsloc/dev_$devname.features.$featureset"
devresults="$resultsloc/dev_$devname.results.$featureset"

# Preprocessing
echo "Preprocessing..."
if [[ $pre == *"spell"* ]]
then
    echo "Running spelling correction..."
    if [ ! -f "$dataloc/train_${trainname}_spell.xml" ]; then
	python spellcorrect.py --in $trainset --out "$dataloc/train_${trainname}_spell.xml"
    fi
    if [ ! -f "$dataloc/dev_${devname}_spell.xml" ]; then
	python spellcorrect.py --in $devset --out "$dataloc/dev_${devname}_spell.xml"
    fi

   trainset="$dataloc/train_${trainname}_spell.xml"
   devset="$dataloc/dev_${devname}_spell.xml"
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
if [ ! -f $trainfeatures ]
then
    echo "Extracting features for train..."
    python raw2vector.py --in $trainset --out $trainfeatures --labels $labels --features $features
fi
if [ ! -f $devfeatures ]
then
    echo "Extracting features for dev..."
    python raw2vector.py --in $devset --out $devfeatures --labels $labels --features $features --keys "$trainfeatures.keys"
fi

# Model
echo "Running svm..."
python svm.py --in $trainfeatures --test $devfeatures --out $devresults --labels $labels --model $model
echo "Done"
