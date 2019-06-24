#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Code to run the keyword classifier and create features for CoD classification

# Global imports
import os

# Local imports
import classify_keywords
import kw_tools

# PARAMETERS: EDIT THESE
filepath = '/u/sjeblee/research/data/va/child_keywords'
vectors = '/u/sjeblee/research/vectors/wikipedia-pubmed-and-PMC-w2v.bin'


def main():
    classify_keywords.supervised_classify(trainfile=os.path.join(filepath, 'full_keywords_19Jun2019.csv'), # CSV file
                                          cat_file=os.path.join(filepath, 'categories_fouo.csv'),           # Keyword category names
                                          kw_file=os.path.join(filepath, 'kw_map_all.csv'),                 # Manually mapped keywords
                                          outfile=os.path.join(filepath, 'kw_map_pred_june19.csv'),         # Output keyword mapping
                                          vecfile=vectors)

    kw_tools.create_csv_from_csv(csv_file=os.path.join(filepath, 'full_keywords_19Jun2019.csv'),            # Same CSV file
                                 kw_file=os.path.join(filepath, 'kw_map_pred_june19.csv'),                  # The keyword mapping created above
                                 cat_file=os.path.join(filepath, 'categories_fouo.csv'),                    # Keyword category names
                                 out_file=os.path.join(filepath, 'mds_child_jan2019_keywords_criteria_binary.csv'),  # Output CSV file
                                 tag_neg=True,              # True to tag negations as -1
                                 include_other=False)       # True to include the 'other' keyword category


if __name__ == "__main__": main()
