#!/usr/bin/python
# Script to run all the steps of the Verbal Autopsy pipeline

import argparse
import extract_features
import heidel_tag
import model
import nn
import os
import results_stats
import spellcorrect
import subprocess

from contextlib import contextmanager

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--train', action="store", dest="train")
    argparser.add_argument('-e', '--ex', action="store", dest="experiment")
    argparser.add_argument('-d', '--test', action="store", dest="test")
    argparser.add_argument('-l', '--labels', action="store", dest="labels")
    argparser.add_argument('-m', '--model', action="store", dest="model")
    argparser.add_argument('-v', '--modelname', action="store", dest="modelname")
    argparser.add_argument('-n', '--name', action="store", dest="name")
    #argparser.add_argument('-r', '--result', action="store", dest="prefix")
    argparser.add_argument('-p', '--preprocess', action="store", dest="preprocess")
    argparser.add_argument('-f', '--features', action="store", dest="feaures")
    args = argparser.parse_args()

    if not (args.train and args.model):
        print "usage: python svm.py"
        print "--in [train.features]"
        print "--test [test.features]"
        print "--out [test.results]"
        print "--labels [ICD_cat/Final_code] (optional)"
        print "--features [type/dem/narr_count/narr_vec/narr_tfidf/kw_count/kw_tfidf]"
        print "--model [nn/lstm/svm/rf/nb]"
        print "--modelname [nn1_all] (optional)"
        print "--name [rnn_ngram3]"
        print "--preprocess [spell/heidel] (optional, default: spell)"
        print "--experiment [traintest/hyperopt] (optional, default: traintest)"
        exit()

    # Parameters
    experiment = "traintest"
    if args.experiment:
        experiment = args.experiment
    if experiment == "traintest" and not args.test:
        print "Error: --test [testfile] require for traintest"
        exit(1)

    pre = "spell"
    if args.preprocess:
        pre = args.preprocess

    labels = "ICD_cat"
    if args.labels:
        labels = args.labels

    if args.modelname:
        modelname = args.modelname
    else:
        modelname = args.model

    if experiment == "traintest":
        run(args.model, modelname, args.train, args.test, args.name, pre, labels)
    if experiment == "hyperopt":
        run(args.model, modelname, args.train, args.test, args.name, pre, labels, arg_hyperopt=True)

def run(arg_model, arg_modelname, arg_train, arg_test, arg_name, arg_preprocess, arg_labels, arg_hyperopt=False):        
    trainname = arg_train + "_cat" # all, adult, child, or neonate
    devname = arg_test + "_cat"
    pre = arg_preprocess
    labels = arg_labels
    featureset="narr_count" # Name of the feature set for feature file
    features="type,narr_count" # type, checklist, narr_bow, narr_tfidf, narr_count, narr_vec, kw_bow, kw_tfidf
    modeltype = arg_model # svm, knn, nn, lstm, nb, rf
    modelname = arg_modelname
    #resultsloc_name = arg_name

    # Location of data files
    dataloc="/u/sjeblee/research/va/data/datasets"
    resultsloc="/u/sjeblee/research/va/data/" + arg_name
    heideldir="/u/sjeblee/tools/heideltime/heideltime-standalone"
    scriptdir="/u/sjeblee/research/va/git/verbal-autopsy"

    # Setup
    if not os.path.exists(resultsloc):
        os.mkdir(resultsloc)
    trainset = dataloc + "/train_" + trainname + ".xml"
    devset = dataloc + "/dev_" + devname + ".xml"
    trainfeatures = resultsloc + "/train_" + trainname + ".features." + featureset
    devfeatures = resultsloc + "/dev_" + devname + ".features." + featureset
    devresults = resultsloc + "/dev_" + devname + ".results." + modelname + "." + featureset

    # Preprocessing
    spname = "spell"
    print "Preprocessing..."
    if "spell" in pre:
        trainsp = dataloc + "/train_" + trainname + "_" + spname + ".xml"
        devsp = dataloc + "/dev_" + devname + "_" + spname + ".xml"
        if not os.path.exists(trainsp):
            print "spellcorrect on train data..."
            spellcorrect.run(trainset, trainsp)
        if not os.path.exists(devsp):
            print "spellcorrect on dev data..."
            spellcorrect.run(devset, devsp)

        trainset = trainsp
        devset = devsp

    if "heidel" in pre:
        print "Running Heideltime..."
        with cd(heideldir):
            trainh = dataloc + "/train_" + trainname + "_ht.xml"
            if not os.path.exists(trainh):
                heidel_tag.run(trainset, trainh)
	        fixtags(trainh)
            trainset = trainh
            devh = dataloc + "/dev_" + devname + "_ht.xml"
            if not os.path.exists(devh):
                heidel_tag.run(devset, devh)
	        fixtags(devh)
            devset = devh

    # Feature Extraction
    print "trainfeatures: " + trainfeatures
    print "devfeatures: " + devfeatures
    if not (os.path.exists(trainfeatures) and os.path.exists(devfeatures)):
        print "Extracting features..."
        extract_features.run(trainset, trainfeatures, devset, devfeatures, features, labels)

    # Model
    if arg_hyperopt:
        print "Running hyperopt..."
        model.hyperopt(modeltype, trainfeatures, devfeatures, devresults, resultsloc, labels)
    else:
        print "Creating model..."
        model.run(modeltype, modelname, trainfeatures, devfeatures, devresults, resultsloc, labels)

    # Results statistics
    print "Calculating scores..."
    results_stats.run(devresults, devresults + ".stats")

    print "Done"

@contextmanager
def cd(newdir):
    prevdir = os.getcwd()
    os.chdir(os.path.expanduser(newdir))
    try:
        yield
    finally:
        os.chdir(prevdir)

def fixtags(filename):
    process = subprocess.Popen(['sed', '-i', '-e', 's/&lt;/</g', filename], stdout=subprocess.PIPE)
    output, err = process.communicate()
    process = subprocess.Popen(['sed', '-i', '-e', 's/&gt;/>/g', filename], stdout=subprocess.PIPE)
    output, err = process.communicate()

if __name__ == "__main__":main()
