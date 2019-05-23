#!/usr/bin/python
# Script to run all the steps of the Verbal Autopsy pipeline

import sys
sys.path.append('./temporal')
sys.path.append('./keywords')
import cluster_keywords
import data_util
import extract_features
import heidel_tag
import medttk_tag
import model
#import rebalance
import results_stats
import spellcorrect
import tag_symptoms
import word2vec

import argparse
import os
import subprocess
import svm
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
    argparser.add_argument('-w', '--modelname', action="store", dest="modelname")
    argparser.add_argument('-n', '--name', action="store", dest="name")
    argparser.add_argument('-p', '--preprocess', action="store", dest="preprocess")
    argparser.add_argument('-f', '--features', action="store", dest="features")
    argparser.add_argument('-z', '--featurename', action="store", dest="featurename")
    argparser.add_argument('-b', '--rebalance', action="store", dest="rebalance")
    argparser.add_argument('-v', '--vectors', action="store", dest="vecfile")
    argparser.add_argument('-i', '--prefix', action="store", dest="prefix")
    argparser.add_argument('-a', '--dataloc', action="store", dest="dataloc")
    argparser.add_argument('-y', '--sympfile', action="store", dest="sympfile")
    argparser.add_argument('-c', '--chvfile', action="store", dest="chvfile")
    args = argparser.parse_args()

    if not (args.train and args.name):
        print "usage: python svm.py"
        print "--in [train.features]"
        print "--test [test.features]"
        print "--out [test.results]"
        print "--labels [ICD_cat/ICD_cat_neo/Final_code] (optional)"
        print "--features [type/dem/narr_count/narr_vec/narr_tfidf/kw_count/kw_tfidf/lda/symp_train/narr_medttk_count/keyword_clusters]"
        print "--featurename [feature_set_name]"
        print "--model [nn/lstm/cnn/filternn/svm/rf/nb]"
        print "--modelname [nn1_all] (optional)"
        print "--name [rnn_ngram3]"
        print "--preprocess [spell/heidel/symp/stem/lemma/medttk/kwc] (optional, default: spell)"
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
        n_feats = 350
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
        run(args.model, modelname, args.train, testset, args.features, fn, args.name, pre, labels, arg_dev=dev, arg_hyperopt=False, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, arg_dataset=dataset, arg_rebalance=rebal, arg_vecfile=args.vecfile, arg_dataloc=args.dataloc, arg_prefix=args.prefix, arg_sympfile=args.sympfile, arg_chvfile=args.chvfile)
    elif experiment == "hyperopt":
        run(args.model, modelname, args.train, testset, args.features, fn, args.name, pre, labels, arg_dev=dev, arg_hyperopt=True, arg_dataset=dataset, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, arg_rebalance=rebal, arg_vecfile=args.vecfile, arg_dataloc=args.dataloc, arg_prefix=args.prefix, arg_sympfile=args.sympfile, arg_chvfile=args.chvfile)
    elif experiment == "crossval":
        crossval(models, args.train, args.features, fn, args.name, pre, labels, args.data, arg_vecfile=args.vecfile, arg_dataloc=args.dataloc, arg_prefix=args.prefix, arg_sympfile=args.sympfile, arg_chvfile=args.chvfile)

    end_time = time.time()
    total_time = end_time - start_time
    if total_time > 3600:
        print "Total time: " + str((total_time/60)/60) + " hours"
    else:
        print "Total time: " + str(total_time/60) + " mins"

def crossval(arg_models, arg_train, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dataset="mds+rct", arg_vecfile="", arg_dataloc="/u/sjeblee/research/va/data/datasets", arg_prefix = "/nbb/sjeblee/va/data/",arg_sympfile=None, arg_chvfile=None):
    print "10-fold cross-validation"
    models = arg_models.split(',')
    dataloc = arg_dataloc
    vecfile = None
    joint_training = False
    if arg_train == 'all':
        joint_training = True

    # Records should be one per line, no xml header or footer
    dset = arg_dataset
    datafile_child = dataloc + "/" + dset + "/all_child_cat_spell.txt"
    datafile_neo = dataloc + "/" + dset + "/all_neonate_cat_spell.txt"
    datafile_adult = dataloc + "/" + dset + "/all_adult_cat_spell.txt"
    datafile = dataloc + "/" + dset + "/all_" + arg_train + "_cat_spell.txt"
    train_extra = []

    #if "spell" in arg_preprocess:
    # TODO: check for spell version if spell in preprocessing

    xml_header = '<root>'
    xml_footer = '</root>'

    # Set up file paths
    #datadir = "/u/sjeblee/research/va/data/" + arg_name
    datadir = arg_prefix + "/" + arg_name
    #datapath = datadir + "/" + arg_dataset
    datapath = dataloc + "/" + dset
    #create_datasets = True
    #if os.path.exists(datapath):
    #    print "Data files already exist, re-using them"
    #    #create_datasets = False
    #else:
    #    os.mkdir(datapath)

    create_datasets = False
    # TODO: If dirs exist already, don't recreate the datasets, just re-run the models that don't have output
    if create_datasets:
        main_files = []
        if joint_training:
            main_files = [datafile_adult, datafile_child, datafile_neo]
        else:
            main_files = [datafile]
        train_extra = []
        use_extra_data = False
        # Extra training data
        if use_extra_data and not joint_training:
            print "Loading extra training data..."
            if arg_train == "adult":
                train_extra = train_extra + get_recs(datafile_child)
                train_extra = train_extra + get_recs(datafile_neo)
            elif arg_train == "child":
                train_extra = train_extra + get_recs(datafile_adult)
                train_extra = train_extra + get_recs(datafile_neo)
                #elif arg_train == "neonate":
                #    train_extra = train_extra + get_recs(datafile_child)

            print "train_extra: " + str(len(train_extra))

        for filename in main_files:
            # Read data file
            print "Reading main data file: " + filename
	    #root = etree.parse(filename).getroot()
            data = {}
            datasets = []
            total = 0

            with open(filename, 'r') as f:
                for line in f:
                    #print "line: " + line
                    cat = None
                    child = etree.fromstring(line)
                    node = child.find(arg_labels)
                    if node != None:
                        cat = node.text
                    if not cat in data:
                        data[cat] = []
                    if cat is not None:
                        data[cat].append(line.strip())
                        total = total + 1
	    '''
	    for child in root:
		node = child.find(arg_labels)
		if node != None:
		    cat = node.text
		if not cat in data:
		    data[cat] = []
		if cat is not None:
		    data[cat].append(child)
		    total = total + 1
	    '''
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
                trainset = train_extra
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
                if joint_training:
                    if 'adult' in filename:
                        trainname = "adult_" + str(z)
                    elif 'child' in filename:
                        trainname = "child_" + str(z)
                    elif 'neonate' in filename:
                        trainname = "neonate_" + str(z)
                else:
                    trainname = arg_train + "_" + str(z)
                #trainfile = datapath + "/train_" + trainname +  "_cat.xml"
                #testfile = datapath + "/test_" + trainname + "_cat.xml"
                trainfile = datapath + "/train_"+ trainname + "_cat_spell.xml"
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
    for z in range(10):
        trainname = arg_train + "_" + str(z)
        if joint_training:
            trainname = 'all'
        vec_feats = "narr_vec" in arg_features or "kw_vec" in arg_features
        #vecfile = arg_vecfile

        # Retrain word vectors for this set
        #vec_feats = False
        if vec_feats:
            print "Training word vectors..."
            if joint_training:
                suffix = '_cat_spell'
                # Concatenate training files
                trainfile = datapath + "/train_all_" + str(z) + suffix + ".txt"
                adult_xml = datapath + "/train_adult_" + str(z) + suffix + ".xml"
                adult_file = datapath + "/train_adult_" + str(z) + suffix + ".txt"
                child_xml = datapath + "/train_child_" + str(z) + suffix + ".xml"
                child_file = datapath + "/train_child_" + str(z) + suffix + ".txt"
                neonate_xml = datapath + "/train_neonate_" + str(z) + suffix + ".xml"
                neonate_file = datapath + "/train_neonate_" + str(z) + suffix + ".txt"
                data_util.xml_to_txt(adult_xml)
                data_util.xml_to_txt(child_xml)
                data_util.xml_to_txt(neonate_xml)

                filenames = [adult_file, child_file, neonate_file]
                with open(trainfile, 'w') as outfile:
                    outfile.write(xml_header + "\n")
                    for fname in filenames:
                        with open(fname) as infile:
                            for line in infile:
                                outfile.write(line)
                    outfile.write(xml_footer + "\n")
            else:
                #trainfile = datapath + "/train_" + trainname +  "_cat.xml"
                trainfile = datapath + "/train_"+ trainname + "_cat_spell.xml"

            #TEMP for keyphrase crossval
            #trainfile = datapath + "/train_all_" + str(z) + "_cat.txt"

            dim = 100
            shouldstem = "stem" in arg_preprocess
            name = "ice+medhelp+narr_all_" + str(z)
            vecfile = datapath + '/' + name
            if not os.path.exists(vecfile):
                vecfile = word2vec.run(trainfile, dim, name, stem=shouldstem)
	    # Yoona: No vector file for cross validation. Use original vector file used for train-test instead
	    #vecfile = arg_vecfile

        for m in models:
            name = arg_name + "/" + m + "_" + str(z)
            modelname = m + "_" + str(z) + "_" + arg_train
            if not os.path.exists(name):
                os.makedirs(name)
            print "Running " + name

            # Model parameters
            n_feats = 350
            anova = "f_classif"
            nodes = 297
            if m == "rf":
                n_feats = 414
                anova = "chi2"
            elif m == "svm":
                n_feats = 350
                anova = "chi2"
            elif m == "nn":
                if arg_train == "neonate":
                    #n_feats = 227
                    nodes = 192
                    anova = "chi2"
                else:
                    n_feats = 350

            run(m, modelname, trainname, trainname, arg_features, arg_featurename, name, arg_preprocess, arg_labels, arg_dev=False, arg_hyperopt=False, arg_dataset=dset, arg_n_feats=n_feats, arg_anova=anova, arg_nodes=nodes, arg_dataloc=dataloc, arg_vecfile=vecfile, arg_crossval_num=z, arg_prefix = arg_prefix,arg_sympfile=arg_sympfile, arg_chvfile=arg_chvfile)

def setup(arg_modelname, arg_train, arg_test, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev, arg_dataloc, arg_vecfile, crossval_num=None, arg_prefix="/u/sjeblee/research/va/data", arg_sympfile=None, arg_chvfile=None):
    print "setup prefix: " + arg_prefix
    if crossval_num is not None:
        trainname = arg_train + "_" + str(crossval_num)
        devname = arg_test + "_" + str(crossval_num)
    else:
        trainname = arg_train
        devname = arg_test
    trainname = trainname + "_cat" # all, adult, child, or neonate
    devname = devname + "_cat"
    pre = arg_preprocess
    print "pre: " + pre
    labels = arg_labels
    featureset = arg_featurename # Name of the feature set for feature file
    features = arg_features # type, checklist, narr_bow, narr_tfidf, narr_count, narr_vec, kw_bow, kw_tfidf, symp_train
    #modeltype = arg_model # svm, knn, nn, lstm, nb, rf
    modelname = arg_modelname
    query_vectors = None

    # Location of data files
    dataloc = arg_dataloc
    resultsloc = arg_prefix + "/" + arg_name
    heideldir="/u/sjeblee/tools/heideltime/heideltime-standalone"
    #scriptdir="/u/sjeblee/research/va/git/verbal-autopsy"

    # Setup
    if not os.path.exists(resultsloc):
        os.mkdir(resultsloc)
    #trainset = dataloc + "/train_" + trainname + ".xml"
    trainset = dataloc + "/train_" + trainname + ".xml" # Trying to make spell_symp for all dataset. Make cross-validation faster
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

    # Edit by Yoona
    element = []
    element.append("narrative")

    # Preprocessing
    spname = "spell"
    print "Preprocessing..."
    if "spell" in pre:
        print "Running spelling correction..."
        #trainsp = dataloc + "/train_" + trainname + "_" + spname + ".xml"
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
        if (arg_sympfile is None or arg_chvfile is None):
            print "Symptom files are not provided."
        print "Tagging symptoms..."
        sympname = "symp"
        tagger_name = "tag_symptoms"
        #trainsp = dataloc + "/train_" + trainname + "_" + sympname + ".xml"
        trainsp = dataloc + "/train_" + trainname + "_" + sympname + ".xml"
        devsp = ""
        if arg_dev:
            devsp = dataloc + "/dev_" + devname + "_" + sympname + ".xml"
        else:
            devsp = dataloc + "/test_" + devname + "_" + sympname + ".xml"
        if not os.path.exists(trainsp):
            print "tag_symptoms on train data..."
            tag_symptoms.run(trainset, trainsp, tagger_name, arg_sympfile, arg_chvfile)
            #fixtags(trainsp)
        if not os.path.exists(devsp):
            print "tag_symptoms on test data..."
            tag_symptoms.run(devset, devsp, tagger_name, arg_sympfile, arg_chvfile)
        #    fixtags(devsp)

        trainset = trainsp
        devset = devsp
        devname = devname + "_" + sympname
        trainname = trainname + "_" + sympname
        #element = "narr_symp"
        element.append("narr_symp")

    if "textrank" in pre:
        print "Extract keywords using textrank"
        textrankname = "textrank"
        tagger_name = "textrank"

        trainsp = dataloc + "/train_" + trainname + "_" + textrankname + ".xml"
        devsp = ""
        if arg_dev:
            devsp = dataloc + "/dev_" + devname + "_" + textrankname + ".xml"
        else:
            devsp = dataloc + "/test_" + devname + "_" + textrankname + ".xml"
        if not os.path.exists(trainsp):
            print "Keyword extraction using textrank on train data..."
            tag_symptoms.run(trainset, trainsp, tagger_name)
        if not os.path.exists(devsp):
            print "Keyword extraction using textrank on test data..."
            tag_symptoms.run(devset, devsp, tagger_name)

        trainset = trainsp
        devset = devsp
        devname = devname + "_" + textrankname
        trainname = trainname + "_" + textrankname
        element.append("narr_textrank")

    if "kwc" in pre:
        numc = "50"
        kwname = "kwkm" + str(numc)
        # TODO: move this setup to a function
        trainkw = dataloc + "/train_" + trainname + "_" + kwname + ".xml"
        devkw = ""
        if arg_dev:
            devkw = dataloc + "/dev_" + devname + "_" + kwname + ".xml"
        else:
            devkw = dataloc + "/test_" + devname + "_" + kwname + ".xml"
        if not os.path.exists(trainkw and devkw):
            print "Keyword clustering..."
            #clusterfile = trainkw + ".clusters"
            clusterfile = dataloc + "/train_" + trainname + "_" + kwname + ".clusters"
            cluster_keywords.run(trainkw, clusterfile, arg_vecfile, trainset, devset, devkw, num_clusters=numc)
        trainset = trainkw
        devset = devkw
        devname = devname + "_" + kwname
        trainname = trainname + "_" + kwname
        #query_vectors = eval(open(clusterfile + '.centers', 'r').read())
        #print('Loaded query vectors:', type(query_vectors))

    print("Elements: ")
    print(element)
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
        if arg_vecfile is not None:
            extract_features.run(trainset, trainfeatures, devset, devfeatures, features, labels, stem, lemma, element, arg_vecfile=arg_vecfile)
        else:
            extract_features.run(trainset, trainfeatures, devset, devfeatures, features, labels, stem, lemma, element)
    return trainfeatures, devfeatures, devresults


def run(arg_model, arg_modelname, arg_train, arg_test, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev=True, arg_hyperopt=False, arg_dataset="mds", arg_n_feats=398, arg_anova="f_classif", arg_nodes=297, arg_dataloc="/u/sjeblee/research/va/data/datasets", arg_rebalance="", arg_vecfile=None, arg_crossval_num=None, arg_prefix="/u/sjeblee/research/va/data",arg_sympfile=None,arg_chvfile=None):

    print "run prefix: " + arg_prefix
    dataloc = arg_dataloc + "/" + arg_dataset
    resultsloc = arg_prefix + "/" + arg_name

    # Joint training
    joint = False
    if arg_train == 'all' and arg_test == 'all':
        joint = True

    if joint:
        trainfeatures_adult, devfeatures_adult, devresults_adult = setup(arg_modelname, 'adult', 'adult', arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev, dataloc, arg_vecfile, crossval_num=arg_crossval_num, arg_prefix=arg_prefix, arg_sympfile=arg_sympfile, arg_chvfile=arg_chvfile)
        trainfeatures_child, devfeatures_child, devresults_child = setup(arg_modelname, 'child', 'child', arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev, dataloc, arg_vecfile, crossval_num=arg_crossval_num, arg_prefix=arg_prefix, arg_sympfile=arg_sympfile, arg_chvfile=arg_chvfile)
        trainfeatures_neonate, devfeatures_neonate, devresults_neonate = setup(arg_modelname, 'neonate', 'neonate', arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev, dataloc, arg_vecfile, crossval_num=arg_crossval_num, arg_prefix=arg_prefix, arg_sympfile=arg_sympfile, arg_chvfile=arg_chvfile)
        trainfeatures = [trainfeatures_adult, trainfeatures_child, trainfeatures_neonate]
        devfeatures = [devfeatures_adult, devfeatures_child, devfeatures_neonate]
        devresults = [devresults_adult, devresults_child, devresults_neonate]
    else:
        trainfeatures, devfeatures, devresults = setup(arg_modelname, arg_train, arg_test, arg_features, arg_featurename, arg_name, arg_preprocess, arg_labels, arg_dev, dataloc, arg_vecfile, arg_prefix=arg_prefix, arg_sympfile=arg_sympfile, arg_chvfile=arg_chvfile)

    labels = arg_labels
    modeltype = arg_model # svm, knn, nn, lstm, nb, rf
    modelname = arg_modelname

    # Model
    if arg_hyperopt:
        print "Running hyperopt..."
        model.hyperopt(modeltype, trainfeatures, devfeatures, devresults, resultsloc, labels)
    else:
        print "Creating model..."
        if joint:
            # The feature files here are lists
            model.run_joint(modeltype, modelname, trainfeatures, devfeatures, devresults, resultsloc, labels, arg_n_feats, arg_anova, arg_nodes, arg_rebalance)
        else:
            model.run(modeltype, modelname, trainfeatures, devfeatures, devresults, resultsloc, labels, arg_n_feats, arg_anova, arg_nodes, arg_rebalance)

        # Results statistics
        print "Calculating scores..."
        if joint:
            results_stats.run(devresults[0], devresults[0] + ".stats")
            results_stats.run(devresults[1], devresults[1] + ".stats")
            results_stats.run(devresults[2], devresults[2] + ".stats")
        else:
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
