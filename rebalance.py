#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Extract features from the xml data
# Feature names: keyword_bow, keyword_tfidf, narr_bow, narr_tfidf

import argparse
from imblearn.over_sampling import SMOTE, ADASYN

import data_util

def main():
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--in', action="store", dest="infile")
        argparser.add_argument('--out', action="store", dest="outfile")
        argparser.add_argument('--name', action="store", dest="name")
        args = argparser.parse_args()

        if not (args.infile and args.outfile):
            print('usage: ./extract_features.py --in [file.features] --out [file.features.resampled]')
            exit()

        name = 'adasyn'
        if args.name:
                name = args.name
        run(args.infile, args.outfile, name)

def run(arg_in, arg_out, arg_labels, arg_name="adasyn"):
        matrix = []
        with open(arg_in, 'r') as f:
                for line in f:
                        vector = eval(line)
                        matrix.append(vector)
        dict_keys = matrix[0].keys()
        new_matrix = rebalance_data(matrix, dict_keys, arg_labels, arg_name)

        # Write output to file
        data_util.write_to_file(new_matrix, dict_keys, arg_out)

def rebalance_data(matrix, dict_keys, labelname, rebal_name):
        # Construct feature vectors and label vector
        features = []
        labels = []
        keys = []
        sizes = []
        flag = True
        for row in matrix:
                feats = []
                for key in dict_keys:
                        val = row[key]
                        if key == labelname:
                                labels.append(val)
                        elif key != "MG_ID":
                                if type(val) is list:
                                        feats = feats + val
                                        if flag:
                                                keys.append(key)
                                                sizes.append(len(val))
                                else:
                                        feats.append(val)
                                        if flag:
                                                keys.append(key)
                                                sizes.append(1)
                features.append(feats)
                if flag:
                        flag = False
        new_feats, new_labels = rebalance(features, labels, rebal_name)
        new_matrix = []
        for y in range(len(new_feats)):
                row = new_feats[y]
                label = new_labels[y]
                index = 0
                new_entry = {}
                for x in range(len(sizes)):
                        key = keys[x]
                        size = sizes[x]
                        val = row[index:index+size]
                        index = index+size
                        new_entry[key] = val
                new_entry[labelname] = label
                new_matrix.append(new_entry)
        return new_matrix

def rebalance(features, labels, name="adasyn"):
        # Count the number of samples for each label
        resample_dist = {}
        for lab in labels:
                if lab in resample_dist:
                        resample_dist[lab] = resample_dist[lab] +1
                else:
                        resample_dist[lab] = 1

        # Print original distribution
        print('original distribution:')
        for key in resample_dist.keys():
                print str(key) + " : " + str(resample_dist[key])

        # Smooth the distribution start by doubling the smallest half of the classes
        values = []
        keys = sorted(resample_dist, key=resample_dist.get)
        for key in keys:
                values.append(resample_dist[key])
        max_items = values[-1]
        for x in range(0, 3):
                val = resample_dist[keys[x]]
                new_val = val + (val*(1/(x+1)))
                print('rebalance class', str(keys[x]), ':', str(val), ' -> ', str(new_val))
                resample_dist[keys[x]] = min(new_val, max_items)

        # Print new distribution
        print('new distribution:')
        for key in resample_dist.keys():
                print str(key) + " : " + str(resample_dist[key])

        x_resampled = []
        y_resampled = []
        if name == "smote" or name == "SMOTE":
                print('smote')
                x_resampled, y_resampled = SMOTE(ratio=resample_dist, k_neighbors=3, kind='svm').fit_sample(features, labels)
        else:
                print('adasyn')
                x_resampled, y_resampled = ADASYN(ratio=resample_dist, n_neighbors=3).fit_sample(features, labels)

        class1 = keys[-1]
        remove1 = int(resample_dist[class1]*0.1)
        class2 = keys[-2]
        remove2 = int(resample_dist[class2]*0.1)

        x_final = []
        y_final = []
        print('Rebalanced data:')
        for z in range(len(y_resampled)):
                label = y_resampled[z]
                if (label == class1) and remove1 > 0:
                        remove1 = remove1 -1
                elif (label == class2) and remove2 > 0:
                        remove2 = remove2 -1
                else:
                        y_final.append(label)
                        x_final.append(x_resampled[z])
                        print str(y_resampled[z]) + " : " + str(x_resampled[z])
        return x_resampled, y_resampled
