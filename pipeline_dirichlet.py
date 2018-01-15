#!/usr/bin/python
# Script to run all the steps of the Verbal Autopsy pipeline

import sys
sys.path.append('./temporal')

import argparse
import extract_features_dirichlet
import heidel_tag
import medttk_tag
import model_dirichlet
import nn
import os
import rebalance
import results_stats
import spellcorrect
import subprocess
import svm
import tag_symptoms
import time

from contextlib import contextmanager
from lxml import etree
from numpy.random import dirichlet
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
    argparser.add_argument('-b', '--rebalance', action="store", dest="rebalance")
    args = argparser.parse_args()

    if not (args.train and args.name):
        print "usage: python svm.py"
        print "--in [train.features]"
        print "--test [test.features]"
        print "--out [test.results]"
        print "--labels [ICD_cat/ICD_cat_neo/Final_code] (optional)"
        print "--features [type/dem/narr_count/narr_vec/narr_tfidf/kw_count/kw_tfidf/lda/symp_train/narr_medttk_count]"
        print "--featurename [feature_set_name]"
        print "--model [nn/lstm/svm/rf/nb]"
        print "--modelname [nn1_all] (optional)"
        print "--name [rnn_ngram3]"
        print "--preprocess [spell/heidel/symp/stem/lemma/medttk] (optional, default: spell)"
        print "--ex [traintest/hyperopt] (optional, default: traintest)"
        print "--dataset [mds_one/mds_tr/mds_one+tr] (optional, default: mds_one)"
        print "--rebalance [adasyn/smote](optional, default: no rebalancing"
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

    rebal = ""
    if args.rebalance:
        rebal = args.rebalance

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

    models = "rb,rf,svm,nn"
    if args.model:
        models = args.model

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
            n_feats = 378

    # Temp for RNN
    #nodes = 600

    if experiment == "traintest":
        run(args.model, modelname, args.train, testset, args.features, fn, args.name, pre, labels, arg_dev=dev, arg_hyperopt=False, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, arg_dataset=dataset, arg_rebalance=rebal)
    elif experiment == "hyperopt":
        run(args.model, modelname, args.train, testset, args.features, fn, args.name, pre, labels, arg_dev=dev, arg_hyperopt=True, arg_dataset=dataset)
    elif experiment == "crossval":
        crossval(modelname, models, args.train, args.features, fn, args.name, pre, labels, args.data)

    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 3600:
        print "Total time: " + str((total_time/60)/60) + " hours"
    else:
        print "Total time: " + str(total_time/60) + " mins"

def crossval(arg_modelname, arg_models, arg_train, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dataset="mds_one"):
    print "10-fold cross-validation"
    models = arg_models.split(',')
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

    #if "spell" in arg_preprocess:
    # TODO: check for spell version if spell in preprocessing

    xml_header = '<dataroot xmlns:od="urn:schemas-microsoft-com:officedata" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="Adult_Anonymous_23Sept2016.xsd" generated="2016-09-23T12:57:36">'
    xml_footer = '</dataroot>'

    # Set up file paths
    datadir = "/u/sjeblee/research/va/data/" + arg_name
    datapath = datadir + "/" + arg_dataset
    create_datasets = True
    if os.path.exists(datapath):
        print "Data files already exist, re-using them"
        #create_datasets = False
    else:
        os.mkdir(datapath)

    # DIRICHLET setup
    num_runs = 100
    #cat_list = ['1','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17']
    #prior = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    #if arg_dataset == "neonate":
    print "using neonatal categories"
    cat_list = ['N1', 'N2', 'N3', 'N4', 'N5']
    prior = [1, 1, 1, 1, 1]
    dirichlets = dirichlet(prior, num_runs)

    data = []
    print "Reading data file..."
    total = 0
    with open(datafile, 'r') as f:
        for line in f:
            data.append(line.strip())

    # TODO: If dirs exist already, don't recreate the datasets, just re-run the models that don't have output
    if create_datasets:
        # DIRICHLET runs
        for z in range(num_runs):
            # Split file into 25% and 75%
            shuffle(data)
            trainset = []
            test_data = {}
            num_test = int(len(data)/4)
            print "test set original: " + str(num_test)

            for number in range(len(data)):
                line = data[number]
                if number < num_test:
                    cat = 'N5'
                    child = etree.fromstring(line)
                    node = child.find(arg_labels)
                    if node != None:
                        cat = node.text
                    if not cat in test_data:
                        test_data[cat] = []
                    test_data[cat].append(line)
                else:
                    trainset.append(line)

            print "train set original: " + str(len(trainset))

            # Generate a Dirichlet distribution
            dist = dirichlets[z]
            target_size = {}
            print "test distribution: "
            for x in range(len(prior)):
                # Generate distribution
                prop = dist[x]
                gen_num = int(round(prop * num_test))
                cat = cat_list[x]
                target_size[cat] = gen_num
                print cat + ": " + str(gen_num)

            # Resample the test set to match generated distribution
            testset = []
            for cat in test_data:
                orig_size = len(test_data[cat])
                size = target_size[cat]
                index = 0
                for y in range(size):
                    if index >= orig_size:
                        index = 0
                    testset.append(test_data[cat][index])
                    index = index+1

            shuffle(testset)
            print "test set dirichlet: " + str(len(testset))

            # Write train and test sets to file
            trainname = arg_train + "_" + str(z)
            trainfile = datapath + "/train_" + trainname +  "_cat_spell.xml"
            testfile = datapath + "/test_" + trainname + "_cat_spell.xml"
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
                        n_feats = 378
            
            run(m, modelname, trainname, trainname, arg_features, arg_featurename, name, arg_preprocess, arg_labels, arg_dev=False, arg_hyperopt=False, arg_dataset=dset, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, dataloc=datadir)
        #os.remove(datapath + "/train_all_" + str(z) + "*")
        #os.remove(datapath + "/test_all_" + str(z) + "*")
        # End DIRICHLET

def run(arg_model, arg_modelname, arg_train, arg_test, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev=True, arg_hyperopt=False, arg_dataset="mds", arg_n_feats=398, arg_anova="f_classif", arg_nodes=297, dataloc="/u/sjeblee/research/va/data/datasets", arg_rebalance=""):

    dataloc = dataloc + "/" + arg_dataset
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
    element = "narrative"

    # Preprocessing
    spname = "spell"
    print "Preprocessing..."
    if "spell" in pre:
        print "Running spelling correction..."
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

    if "medttk" in pre:
        print "Running medttk..."
        #with cd(heideldir):
        trainh = dataloc + "/train_" + trainname + "_medttk.xml"
        if not os.path.exists(trainh):
            medttk_tag.run(trainset, trainh)
	    fixtags(trainh)
        trainset = trainh
        devh = ""
        if arg_dev:
            devh = dataloc + "/dev_" + devname + "_medttk.xml"
        else:
            devh = dataloc + "/test_" + devname + "_medttk.xml"
        if not os.path.exists(devh):
            medttk_tag.run(devset, devh)
	    fixtags(devh)
        devset = devh
        devname = devname + "_medttk"
        trainname = trainname + "_medttk"
        element = "narr_medttk"

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
    if arg_dev:
        devset = dataloc + "/dev_" + devname + ".xml"
        devfeatures = dataloc + "/dev_" + devname + ".features." + featureset
        devresults = resultsloc + "/dev_" + devname + ".results." + modelname + "." + featureset
    else:
        devset = dataloc + "/test_" + devname + ".xml"
        devfeatures = dataloc + "/test_" + devname + ".features." + featureset
        devresults = resultsloc + "/test_" + devname + ".results." + modelname + "." + featureset
    trainfeatures = dataloc + "/train_" + trainname + ".features." + featureset
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
        extract_features_dirichlet.run(trainset, trainfeatures, devset, devfeatures, features, labels, stem, lemma, element)

    # Rebalance dataset?
    if arg_rebalance != "":
        rebalancedfeatures = trainfeatures + "." + arg_rebalance
        rebalance.run(trainfeatures, rebalancedfeatures, labels, arg_rebalance)
        trainfeatures = rebalancedfeatures

    # Model
    if arg_hyperopt:
        print "Running hyperopt..."
        model_dirichlet.hyperopt(modeltype, trainfeatures, devfeatures, devresults, resultsloc, labels)
    else:
        print "Creating model..."
        if modeltype == "nb":
            svm.run(modeltype, trainfeatures, devfeatures, devresults, labels)
        #elif modeltype == "cnn":
        #    model_temp.run(modeltype, modelname, trainfeatures, devfeatures, devresults, resultsloc, labels, arg_n_feats, arg_anova, arg_nodes)
        else:
            model_dirichlet.run(modeltype, modelname, trainfeatures, devfeatures, devresults, resultsloc, labels, arg_n_feats, arg_anova, arg_nodes)

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
    # TODO: do we need this?
    #del recs[len(recs)-1]
    #del recs[0]
    return recs

if __name__ == "__main__":main()
