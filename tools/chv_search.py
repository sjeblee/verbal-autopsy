#!/usr/bin/python
# -*- coding: utf-8 -*-

def main():
    # Import CHV file
    chv_terms = []
    chv_file = "../../res/CHV_concepts.csv"
    narr_file = "/u/sjeblee/research/va/data/datasets/all_narrsent_space.txt"

    with open(chv_file, 'r') as f:
        for line in f.readlines():
            items = line.split('\t')
            term = items[1]
            mapped_term = items[2]
            vector = [term, mapped_term, 0]
            chv_terms.append(vector)
    print "chv terms: " + str(len(chv_terms))
        
    # Search for terms in the narratives
    for x in range(len(chv_terms)):
        search_term = " " + chv_terms[x][0] + " "
        count = open(narr_file, 'r').read().count(search_term)
        chv_terms[x][2] = count
        if count > 0:
            print chv_terms[x][0] + "\t" + chv_terms[x][1] + "\t" + str(count)

if __name__ == "__main__":main()
