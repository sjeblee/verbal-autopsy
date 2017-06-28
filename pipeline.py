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
import svm
import tag_symptoms
import time

from contextlib import contextmanager
from lxml import etree
from random import shuffle

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-t', '--train', action="store", dest="train")
    argparser.add_argument('-e', '--ex', action="store", dest="experiment")
    argparser.add_argument('-d', '--dev', action="store", dest="dev")
    argparser.add_argument('-s', '--test', action="store", dest="test")
    argparser.add_argument('-g', '--dataset', action="store", dest="data")
    argparser.add_argument('-l', '--labels', action="store", dest="labels")
    argparser.add_argument('-m', '--model', action="store", dest="model")
    argparser.add_argument('-v', '--modelname', action="store", dest="modelname")
    argparser.add_argument('-n', '--name', action="store", dest="name")
    argparser.add_argument('-p', '--preprocess', action="store", dest="preprocess")
    argparser.add_argument('-f', '--features', action="store", dest="features")
    argparser.add_argument('-z', '--featurename', action="store", dest="featurename")
    args = argparser.parse_args()

    if not (args.train and args.name):
        print "usage: python svm.py"
        print "--in [train.features]"
        print "--test [test.features]"
        print "--out [test.results]"
        print "--labels [ICD_cat/Final_code] (optional)"
        print "--features [type/dem/narr_count/narr_vec/narr_tfidf/kw_count/kw_tfidf/lda/symp_train]"
        print "--featurename [feature_set_name]"
        print "--model [nn/lstm/svm/rf/nb]"
        print "--modelname [nn1_all] (optional)"
        print "--name [rnn_ngram3]"
        print "--preprocess [spell/heidel/symp/stem/lemma] (optional, default: spell)"
        print "--ex [traintest/hyperopt] (optional, default: traintest)"
        print "--dataset [mds_one/mds_tr/mds_one+tr] (optional, default: mds_one)"
        exit()

    # Timing
    start_time = time.time()

    # Parameters
    experiment = "traintest"
    if args.experiment:
        experiment = args.experiment
    if experiment == "traintest" and not (args.test or args.dev):
        print "Error: --test [testfile] or --dev [devfile] required for traintest"
        exit(1)
    elif not experiment == "crossval" and not args.model:
        print "Error: --model [modeltype] required"

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

    dev = True
    if args.test:
        testset = args.test
        dev = False
    else:
        testset = args.dev

    dataset = "mds_one"
    if args.data:
        dataset = args.data

    fn = "feats"
    if args.featurename:
        fn = args.featurename

    # Model parameters
    n_feats = 200
    anova = "f_classif"
    nodes = 297
    if args.model == "rf":
        n_feats = 414
        anova = "chi2"
    elif args.model == "svm":
        n_feats = 378
        anova = "chi2"
    elif args.model == "nn":
        if args.train == "neonate" or args.train == "child_neo":
            n_feats = 227
            nodes = 192
            anova = "chi2"
        else:
            n_feats = 398

    if experiment == "traintest":
        run(args.model, modelname, args.train, testset, args.features, fn, args.name, pre, labels, arg_dev=dev, arg_hyperopt=False, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, arg_dataset=dataset)
    elif experiment == "hyperopt":
        run(args.model, modelname, args.train, testset, args.features, fn, args.name, pre, labels, arg_dev=dev, arg_hyperopt=True, arg_dataset=dataset)
    elif experiment == "crossval":
        crossval(modelname, args.train, args.features, fn, args.name, pre, labels, args.data)

    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 3600:
        print "Total time: " + str((total_time/60)/60) + " hours"
    else:
        print "Total time: " + str(total_time/60) + " mins"

def crossval(arg_modelname, arg_train, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dataset="mds_one"):
    print "10-fold cross-validation"
    models = ['nb', 'rf', 'svm', 'nn']
    dataloc = "/u/sjeblee/research/va/data/datasets"

    # Records should be one per line, no xml header or footer
    dset = arg_dataset
    datafile_child = dataloc + "/" + dset + "/all_child_cat_spell.txt"
    datafile_neo = dataloc + "/" + dset + "/all_neonate_cat_spell.txt"
    datafile_adult = dataloc + "/" + dset + "/all_adult_cat_spell.txt"
    datafile = dataloc + "/" + dset + "/all_" + arg_train + "_cat_spell.txt"
    records = []
    data = {}
    datasets = []
    train_extra = []

    # TODO: check for spell version if spell in preprocessing

    xml_header = '<dataroot xmlns:od="urn:schemas-microsoft-com:officedata" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="Adult_Anonymous_23Sept2016.xsd" generated="2016-09-23T12:57:36">'
    xml_footer = '</dataroot>'

    # Extra training data
    print "Loading extra training data..."
    if arg_train == "adult":
        train_extra = train_extra + get_recs(datafile_child)
        train_extra = train_extra + get_recs(datafile_neo)
    elif arg_train == "child":
        train_extra = train_extra + get_recs(datafile_adult)
        train_extra = train_extra + get_recs(datafile_neo)
    elif arg_train == "neonate":
        train_extra = train_extra + get_recs(datafile_child)

    print "train_extra: " + str(len(train_extra))

    # Read data file
    print "Reading main data file..."
    #tree = etree.parse(datafile)
    #root = tree.getroot()
    total = 0
    with open(datafile, 'r') as f:
        for line in f:
            cat = 15
            child = etree.fromstring(line)
            node = child.find("ICD_cat")
            if node != None:
                cat = int(node.text)
            if not cat in data:
                data[cat] = []
            data[cat].append(line.strip())
            total = total + 1
    print "Records in main data file: " + str(total)

    # Determine how many records we need from each category
    num = {}
    for category in data:
        recs = data[category]
        n = len(recs)/10
        shuffle(data[category])
        if n == 0:
            n = 1
        num[category] = n
    print "Num of recs from each cat: " + str(num)

    for x in range(10):
        # Construct datasets
        print "Constructing dataset " + str(x)
        recset = []
        for category in data:
            numrecs = num[category]

            # Add recs to the set and remove from original list
            if len(data[category]) > numrecs:
                recset = recset + data[category][0:numrecs]
                del data[category][0:numrecs]
            elif len(data[category]) > 0:
                recset = recset + data[category]
                data[category] = []
            else:
                print "no recs added from cat " + str(category) + " because it was empty"
        datasets.append(recset)
        print "total recs: " + str(len(recset))

    for z in range(10):
        # Construct train and test sets
        testset = datasets[z]
        trainset = [] + train_extra
        if z > 0:
            for u in range(0, z):
                trainset = trainset + datasets[u]
        if z < 9:
            for v in range(z+1, 10):
                trainset = trainset + datasets[v]

        shuffle(trainset)
        shuffle(testset)

        print "Train: " + str(len(trainset))
        print "Test: " + str(len(testset))

        # Write train and test sets to file
        trainname = arg_train + "_" + str(z)
        datadir = "/u/sjeblee/research/va/data/" + arg_name
        trainfile = datadir + "/train_" + trainname +  "_cat_spell.xml"
        testfile = datadir + "/test_" + trainname + "_cat_spell.xml"
        outfile = open(trainfile, 'w')
        outfile.write(xml_header + "\n")
        for item in trainset:
            outfile.write(item + "\n")
        outfile.write(xml_footer + "\n")
        outfile.close()

        outfile2 = open(testfile, 'w')
        outfile2.write(xml_header + "\n")
        for item in testset:
            outfile2.write(item + "\n")
        outfile2.write(xml_footer + "\n")
        outfile2.close()

        # Run models
        for m in models:
            name = arg_name + "/" + m + "_" + str(z)
            modelname = m + "_" + str(z) + "_" + arg_train
            if not os.path.exists(name):
                os.makedirs(name)
            print "Running " + name

            # Model parameters
            n_feats = 200
            anova = "f_classif"
            nodes = 297
            if m == "rf":
                n_feats = 414
                anova = "chi2"
            elif m == "svm":
                n_feats = 378
                anova = "chi2"
            elif m == "nn":
                if arg_train == "neonate":
                    n_feats = 227
                    nodes = 192
                    anova = "chi2"
                else:
                    n_feats = 398
            
            run(m, modelname, trainname, trainname, arg_features, arg_featurename, name, arg_preprocess, arg_labels, arg_dev=False, arg_hyperopt=False, arg_dataset=dset, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, dataloc=datadir)

def run(arg_model, arg_modelname, arg_train, arg_test, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev=True, arg_hyperopt=False, arg_dataset="mds_one", arg_n_feats=398, arg_anova="f_classif", arg_nodes=297, dataloc="/u/sjeblee/research/va/data/datasets"):

    #dataloc = dataloc + "/" + arg_dataset
    trainname = arg_train + "_cat" # all, adult, child, or neonate
    devname = arg_test + "_cat"
    pre = arg_preprocess
    print "pre: " + pre
    labels = arg_labels
    featureset = arg_featurename # Name of the feature set for feature file
    features = arg_features # type, checklist, narr_bow, narr_tfidf, narr_count, narr_vec, kw_bow, kw_tfidf, symp_train
    modeltype = arg_model # svm, knn, nn, lstm, nb, rf
    modelname = arg_modelname
    #resultsloc_name = arg_name

    # Location of data files
    #dataloc="/u/sjeblee/research/va/data/datasets"
    resultsloc="/u/sjeblee/research/va/data/" + arg_name
    heideldir="/u/sjeblee/tools/heideltime/heideltime-standalone"
    scriptdir="/u/sjeblee/research/va/git/verbal-autopsy"

    # Setup
    if not os.path.exists(resultsloc):
        os.mkdir(resultsloc)
    trainset = dataloc + "/train_" + trainname + ".xml"
    devset = ""
    devfeatures = ""
    devresults = ""
    if arg_dev:
        devset = dataloc + "/dev_" + devname + ".xml"
        devfeatures = dataloc + "/dev_" + devname + ".features." + featureset
        devresults = resultsloc + "/dev_" + devname + ".results." + modelname + "." + featureset
    else:
        devset = dataloc + "/test_" + devname + ".xml"
        devfeatures = dataloc + "/test_" + devname + ".features." + featureset
        devresults = resultsloc + "/test_" + devname + ".results." + modelname + "." + featureset
    trainfeatures = dataloc + "/train_" + trainname + ".features." + featureset

    # Preprocessing
    spname = "spell"
    print "Preprocessing..."
    if "spell" in pre:
        trainsp = dataloc + "/train_" + trainname + "_" + spname + ".xml"
        devsp = ""
        if arg_dev:
            devsp = dataloc + "/dev_" + devname + "_" + spname + ".xml"
        else:
            devsp = dataloc + "/test_" + devname + "_" + spname + ".xml"
        if not os.path.exists(trainsp):
            print "spellcorrect on train data..."
            spellcorrect.run(trainset, trainsp)
        if not os.path.exists(devsp):
            print "spellcorrect on test data..."
            spellcorrect.run(devset, devsp)

        trainset = trainsp
        devset = devsp
        devname = devname + "_" + spname
        trainname = trainname + "_" + spname

    if "heidel" in pre:
        print "Running Heideltime..."
        with cd(heideldir):
            trainh = dataloc + "/train_" + trainname + "_ht.xml"
            if not os.path.exists(trainh):
                heidel_tag.run(trainset, trainh)
	        fixtags(trainh)
            trainset = trainh
            devh = ""
            if arg_dev:
                devh = dataloc + "/dev_" + devname + "_ht.xml"
            else:
                devh = dataloc + "/test_" + devname + "_ht.xml"
            if not os.path.exists(devh):
                heidel_tag.run(devset, devh)
	        fixtags(devh)
            devset = devh
        devname = devname + "_ht"
        trainname = trainname + "_ht"

    if "symp" in pre:
        print "Tagging symptoms..."
        sympname = "symp"
        trainsp = dataloc + "/train_" + trainname + "_" + sympname + ".xml"
        #devsp = ""
        #if arg_dev:
        #    devsp = dataloc + "/dev_" + devname + "_" + sympname + ".xml"
        #else:
        #    devsp = dataloc + "/test_" + devname + "_" + sympname + ".xml"
        if not os.path.exists(trainsp):
            print "tag_symptoms on train data..."
            tag_symptoms.run(trainset, trainsp)
            #fixtags(trainsp)
        #if not os.path.exists(devsp):
        #    print "tag_symptoms on test data..."
        #    tag_symptoms.run(devset, devsp)
        #    fixtags(devsp)

        trainset = trainsp
        #devset = devsp
        #devname = devname + "_" + spname
        trainname = trainname + "_" + spname

    # Feature Extraction
    print "trainfeatures: " + trainfeatures
    print "devfeatures: " + devfeatures
    stem = False
    lemma = False
    if "stem" in pre:
        stem = True
    if "lemma" in pre:
        lemma = True
    print "stem: " + str(stem) + " lemma: " + str(lemma)
    if not (os.path.exists(trainfeatures) and os.path.exists(devfeatures)):
        print "Extracting features..."
        extract_features.run(trainset, trainfeatures, devset, devfeatures, features, labels, stem, lemma)

    # Model
    if arg_hyperopt:
        print "Running hyperopt..."
        model.hyperopt(modeltype, trainfeatures, devfeatures, devresults, resultsloc, labels)
    else:
        print "Creating model..."
        if modeltype == "nb":
            svm.run(modeltype, trainfeatures, devfeatures, devresults, labels)
        else:
            model.run(modeltype, modelname, trainfeatures, devfeatures, devresults, resultsloc, labels, arg_n_feats, arg_anova, arg_nodes)

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

def get_recs(filename):
    recs = []
    with open(filename, 'r') as f:
        for line in f:
            recs.append(line.strip())
    del recs[len(recs)-1]
    del recs[0]
    return recs

if __name__ == "__main__":main()
