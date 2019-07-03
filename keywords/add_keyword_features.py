#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Code to run the keyword classifier and create features for CoD classification

# Global imports
import os

# Local imports
#import classify_keywords
import cluster_keywords
import kw_tools

# PARAMETERS: EDIT THESE
filepath = '/u/sjeblee/research/data/va/child_keywords'
vectors = '/u/sjeblee/research/vectors/wikipedia-pubmed-and-PMC-w2v.bin'


def main():
    # Uncomment this block for supervised classifier
    #classify_keywords.supervised_classify(trainfile=os.path.join(filepath, 'full_keywords_19Jun2019.csv'), # CSV file
    #                                      cat_file=os.path.join(filepath, 'categories_fouo.csv'),           # Keyword category names
    #                                      kw_file=os.path.join(filepath, 'kw_map_all.csv'),                 # Manually mapped keywords
    #                                      outfile=os.path.join(filepath, 'kw_map_pred_july2_elmo.csv'),         # Output keyword mapping
    #                                      vecfile=None,                                                     # filepath for pubmed vectors if using pubmed model, None for Elmo model
    #                                      model_type='elmo',                                                # 'pubmed' or 'elmo'
    #                                      eval=False)                                                       # True to use 10% of training data for evaluation

    # OR - uncomment this block for unsupervised clustering of keywords
    cluster_keywords.unsupervised_cluster(infile=os.path.join(filepath, 'full_keywords_19Jun2019.csv'), # CSV file
                                          clusterfile=os.path.join(filepath, 'kw_clusters_43.csv'),        # Choose a name for the cluster output file
                                          outfile=os.path.join(filepath, 'kw_map_clusters_43.csv'),        # Choose a name for the output keyword mapping
                                          vecfile=vectors,                                                  # File containing word embeddings
                                          num_clusters=43)                                                  # The number of clusters to generate

    # Use either the supervised or unsupervised keyword groupings to generate binary features for each record
    kw_tools.create_csv_from_csv(csv_file=os.path.join(filepath, 'full_keywords_19Jun2019.csv'),            # Same CSV file
                                 kw_file=os.path.join(filepath, 'kw_map_clusters_43.csv'),        # The keyword mapping created above
                                 cat_file=os.path.join(filepath, 'kw_clusters_43.names'),                    # Keyword category names
                                 out_file=os.path.join(filepath, 'mds_child_jan2019_keywords_clusters43.csv'),  # Output CSV file
                                 include_other=True,       # True to include the 'other' keyword category
                                 tag_neg=False,             # True to use negex to tag negative keywords
                                 num_cats=43)               # Number of keyword categories (should be the same as num_clusters if using unsupervised)


if __name__ == "__main__": main()
