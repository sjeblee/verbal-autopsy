#!/usr/bin/python
# Build an SVM classifier with the VA features
# @author sjeblee@cs.toronto.edu

import argparse
import time
from sklearn import metrics
from sklearn import neighbors
from sklearn import preprocessing
from sklearn import svm
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline

labelencoder = None

def main():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--in', action="store", dest="infile")
    argparser.add_argument('--out', action="store", dest="outfile")
    argparser.add_argument('--test', action="store", dest="testfile")
    argparser.add_argument('--labels', action="store", dest="labelname")
    argparser.add_argument('--model', action="store", dest="model")
    args = argparser.parse_args()

    if not (args.infile and args.outfile and args.testfile):
        print "usage: python svm.py --in [train.features] --test [test.features] --output [test.results] --labels [ICD_cat/Final_code] --model [svm/knn]"
        exit()

    global labelname
    labelname = "Final_code"
    if args.labelname:
        labelname = args.labelname
    model = "svm"
    if args.model:
        model = args.model
        
    trainids = []             # VA record id
    trainlabels = []   # Correct ICD codes
    X = []               # Feature vectors
    Y = []
    
    # Read in feature keys
    global keys
    with open(args.infile + ".keys", "r") as kfile:
        keys = eval(kfile.read())
        
    global labelencoder
    labelencoder = preprocessing.LabelEncoder() # Transforms ICD codes to numbers
    Y = preprocess(args.infile, trainids, trainlabels, X, Y, True)
    print "X: " + str(len(X)) + "\nY: " + str(len(Y))

    # Train model
    print "training model..."
    
    anova_filter = SelectKBest(f_classif, k=200)
    if model == "svm":
        clf = svm.SVC(kernel='rbf')
        pipeline = make_pipeline(anova_filter, clf)
    elif model == "knn":
        print "k-nearest neighbor"
        knn = neighbors.KNeighborsClassifier(n_neighbors=5, weights='distance', n_jobs=-1)
        pipeline = make_pipeline(anova_filter, knn)
        
    pipeline.fit(X, Y)

    selected = anova_filter.get_support(True)
    print "features selected: "
    for i in selected:
        print "\t" + keys[i+2]

    # Test
    print "testing..."
    testids = []
    testlabels = []
    testX = []
    testY = []
    testY = preprocess(args.testfile, testids, testlabels, testX, testY)
    results = pipeline.predict(testX)
    predictedlabels = labelencoder.inverse_transform(results)

    # TODO: calculate F1 score of results
    print "calculating scores..."
    precision = metrics.precision_score(testY, results)
    print "precision: " + str(precision)
    recall = metrics.recall_score(testY, results)
    print "recall: " + str(recall)
    f1score = metrics.f1_score(testY, results)
    print "f1: " + str(f1score)

    # Write results to a file
    output = open(args.outfile, 'w')
    for i in range(len(testids)):
        out = {}
        out['MG_ID'] = testids[i]
        out['Correct_ICD'] = testlabels[i]
        out['Predicted_ICD'] = predictedlabels[i]
        output.write(str(out) + "\n")
    output.close()

def preprocess(filename, ids, labels, x, y, trainlabels=False):
    global labelencoder
    # Read in the feature vectors
    starttime = time.time()
    print "preprocessing features..."
    with open(filename, 'r') as f:
        for line in f:
            vector = eval(line)
            features = []
            for key in keys:
                if key == 'MG_ID':
                    ids.append(vector[key])
                    #print "ID: " + vector[key]
                elif key == labelname:
                    labels.append(vector[key])
                else:
                    if vector.has_key(key):
                        features.append(vector[key])
                    else:
                        features.append('0')
            x.append(features)
            #print "features: " + str(features)
    print "trainlabels: " + str(labels)

    # Convert ICD codes to numerical labels
    if trainlabels:
        labelencoder.fit(labels)
        #print "encoded labels: " + str(labelencoder.classes_)
    y = labelencoder.transform(labels)
    # Normalize features to 0 to 1
    preprocessing.minmax_scale(x, copy=False)
    endtime = time.time()
    print "preprocessing took " + str(endtime - starttime) + " s"
    return y


if __name__ == "__main__":main() 
