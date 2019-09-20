#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Cluster the keywords from the records using word2vec

from gensim.models import KeyedVectors, Word2Vec
from lxml import etree
import csv
import pandas

#sys.path.append('/home/sjeblee/Documents/Research/git/negex/negex.python') # PATH to NEGEX
from negex.negex import negTagger, sortRules


''' Create a csv file from an xml file
    rec_file: The xml file
    cat_file: The ICD category mapping file
    out_file: The csv file to write to
'''
def create_binary_csv(rec_file, cat_file, out_file):
    cat_map = load_keyword_map(cat_file)
    tree = etree.parse(rec_file)
    fieldnames = ['ID', 'COD_CATEGORY', 'Codex_WBD10_Child']
    for num in range(0, 44):
        fieldnames.append(str(num))

    with open(out_file, 'w') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for child in tree.getroot():
            recid = child.find('MG_ID').text
            cat = child.find('cghr_cat').text
            icd = ''
            icd_node = child.find('Codex_WBD10_Child')
            if icd_node is not None:
                icd = icd_node.text
            # Get the keywords and normalize them
            keywords = get_keywords(child).replace(';', ',')
            keywords = keywords.replace('|', ',')
            keywords_fixed = []
            for kw_phrase in keywords.strip().split(','):
                keywords_fixed.append(kw_phrase)

            # Initialize the row values
            row = {}
            for fn in fieldnames:
                row[fn] = 0
            row['ID'] = recid
            row['COD_CATEGORY'] = cat
            row['Codex_WBD10_Child'] = icd
            for kw in keywords_fixed:
                if kw in cat_map:
                    kw_cat = cat_map[kw]
                else:
                    kw_cat = '43'
                row[kw_cat] = 1
            writer.writerow(row)


''' Add the keyword category features to an existing csv file *USE THIS*
    csv_file: The original csv file
    cat_file: The mapping of keyword category numbers to names
    out_file: The csv file to write to (will include all orginal columns plus the new features)
    tag_neg: True to detect negations and set values to -1, False for binary features
    include_other: True to include the 'Other' keyword class and corresponding features, False to exclude it
'''
def create_csv_from_csv(csv_file, kw_file, cat_file, out_file, tag_neg=False, include_other=False, use_criteria=False, num_cats=43):
    print('create csv from csv')
    #mapfile = '/home/sjeblee/Documents/Research/VerbalAutopsy/data/category_maps/cghr_code_map_child.csv'
    #icdmap = data_util.get_icd_map(mapfile)

    cat_map = load_keyword_map(kw_file)
    df = pandas.read_csv(csv_file)
    df = df.fillna('')

    cat_names = load_category_map(cat_file)
    if include_other:
        cat_names[0] = 'other'

    # Load negex rules
    if tag_neg:
        rfile = open('negex/negex_triggers.txt', 'r')
        irules = sortRules(rfile.readlines())

    # Set default values
    df['COD_CATEGORY'] = ''
    df['keywords_fixed'] = ''
    if use_criteria:
        df['criteria_diarrhea'] = '0'
        df['criteria_fouo'] = '0'
        df['criteria_malaria'] = '0'
        df['criteria_meningitis_encephalitis'] = '0'
        df['criteria_pneumonia'] = '0'
        df['criteria_tb'] = '0'

    for num in range(1, num_cats):
        kw_name = 'kw_' + cat_names[int(num)]
        df[kw_name] = '0'
    if include_other:
        kw_name = 'kw_' + cat_names[0]
        df[kw_name] = '0'

    # Add keyword pairs
    max_num = num_cats
    min_num = 1
    if include_other:
        min_num = 0
    for num in range(min_num, max_num):
        for num2 in range(min_num, max_num):
            if num < num2:
                name1 = cat_names[int(num)]
                name2 = cat_names[int(num2)]
                df['kw2_' + name1 + '_' + name2] = '0'

    for i, row in df.iterrows():
        # Get ICD code and map it to cghr_cat
        '''
        code = df.at[i, 'final_code']
        if code in icdmap:
            cat = str(icdmap[code])
        else:
            cat = '9' # Other category
        df.at[i, 'COD_CATEGORY'] = cat
        '''

        # Get the keywords and normalize them
        kw1 = df.at[i, 'p1_keywords']
        kw2 = df.at[i, 'p2_keywords']
        #print('kw1:', kw1, 'kw2:', kw2)
        keywords = kw1 + kw2
        #keywords = df.at[i, 'Terms']
        keywords = clean_keywords(keywords)
        keywords_fixed = []
        print('keywords:', keywords)

        for kw_phrase in keywords.strip().split(','):
            kw_phrase = kw_phrase.strip()
            keywords_fixed.append(kw_phrase)
            kw_class = map_keyword(kw_phrase, cat_map)
            kw_name = 'kw_' + cat_names[int(kw_class)]
            val = '1'

            # Negation detection
            if tag_neg:
                words = kw_phrase.split(' ')
                phrase_words = []
                for word in words:
                    word = word.strip()
                    if len(word) > 0:
                        phrase_words.append(word)
                print('phrase_words:', phrase_words)

                tagger = negTagger(sentence=kw_phrase, phrases=phrase_words, rules=irules, negP=False)
                neg_sent = tagger.getNegTaggedSentence()
                neg_flag = tagger.getNegationFlag()
                scopes = tagger.getScopes()
                print('negex on', kw_phrase)
                print('neg_sent:', neg_sent)
                print('scopes:', scopes)
                print('neg_flag:', neg_flag)
                # If phrase is neg, set val to -1
                if neg_flag == 'negated':
                    val = '-1'

            # Save the val to the dataframe
            df.at[i, kw_name] = val
        df.at[i, 'keywords_fixed'] = ','.join(keywords_fixed)

        v = {}
        for num in range(1, max_num):
            kw_name = 'kw_' + cat_names[int(num)]
            v[num] = (int(df.at[i, kw_name]) == 1)

        # Add diagnostic criteria
        if use_criteria:

            if v[1] or v[2] or v[3]:
                df.at[i, 'criteria_fouo'] = '1'
            if v[7] and (v[1] or v[2] or v[3]) and (v[13] or v[14] or v[6] or v[20] or v[40]):
                df.at[i, 'criteria_tb'] = '1'
            if v[7] and v[2] and (v[6] or v[9] or v[8] or v[14] or v[13]) and not (v[11] or v[27] or v[25]):
                df.at[i, 'criteria_pneumonia'] = '1'
            if v[24] and (v[31] or v[28] or v[1] or v[2] or v[3]):
                df.at[i, 'criteria_diarrhea'] = '1'
            if v[2] and v[19] and (v[26] or v[17] or v[18] or v[31] or v[6]):
                df.at[i, 'criteria_malaria'] = '1'
            if ((v[1] or v[2] or v[3]) and v[19] and not (v[6] or v[7] or v[8] or v[9] or v[10] or v[11] or v[12])) or (v[37] or ((v[17] or v[36]) and (v[31] or v[18] or v[19]))):
                df.at[i, 'criteria_meningitis_encephalitis'] = '1'

        # Add keyword pairs
        for num in range(1, max_num):
            for num2 in range(1, max_num):
                if num < num2:
                    if v[num] and v[num2]:
                        name1 = cat_names[int(num)]
                        name2 = cat_names[int(num2)]
                        df.at[i, 'kw2_' + name1 + '_' + name2] = '1'

    # Write the file
    df.to_csv(out_file)


''' Pre-process keyword string: Replaces [double space, ; | & and] with , for separating keywords
    keywords: the keyword string to process
    returns: a string
'''
def clean_keywords(keywords):
    keywords = keywords.lower()
    keywords = keywords.replace('  ', ',').replace(';', ',').replace('|', ',').replace('.', ',').replace('/', ',')
    keywords = keywords.replace('&', ',').replace(' and ', ',') # Fix and separators
    return keywords


def extract_keyword_list(csvfile, outfile):
    df = pandas.read_csv(csvfile)
    keywords = []
    for i, row in df.iterrows():
        kw_string = str(row['keywords_fixed'])
        for kw in kw_string.split(','):
            kw = kw.strip()
            if len(kw) > 0 and kw not in keywords:
                keywords.append(kw)
    outf = open(outfile, 'w')
    for kw in keywords:
        outf.write(kw + '\n')
    outf.close()


def get_keywords_from_csv(csvfile):
    df = pandas.read_csv(csvfile)
    keywords_fixed = []
    for i, row in df.iterrows():
        # Get the keywords and normalize them
        kw1 = str(df.at[i, 'p1_keywords'])
        kw2 = str(df.at[i, 'p2_keywords'])
        #print('kw1:', kw1, 'kw2:', kw2)
        keywords = kw1 + kw2
        #keywords = df.at[i, 'Terms']
        keywords = clean_keywords(keywords)
        print('keywords:', keywords)

        for kw_phrase in keywords.strip().split(','):
            kw_phrase = kw_phrase.strip()
            keywords_fixed.append(kw_phrase)
    return keywords_fixed


'''
   Get the physician-generated keyword from a record
   elem: the xml element representing the record
'''
def get_keywords(elem, name=None):
    keyword_string = ""
    if name is None:
        keywords1 = elem.find('CODINGKEYWORDS1')
        if keywords1 is not None and keywords1.text is not None:
            keyword_string = keyword_string + keywords1.text.encode("utf-8")
        keywords2 = elem.find('CODINGKEYWORDS2')
        if keywords2 is not None and keywords2.text is not None:
            if keyword_string != "":
                keyword_string = keyword_string + ","
            keyword_string = keyword_string + keywords2.text.encode("utf-8")
    else:
        keywords = elem.find(name)
        if keywords is not None and keywords.text is not None:
            keyword_string = keywords.text.encode("utf-8")

    return keyword_string.lower()


''' Load the keyword category map from a csv file
    filename: The csv file to load from
'''
def load_category_map(filename):
    cat_map = {}
    df = pandas.read_csv(filename)
    for i, row in df.iterrows():
        cat = int(row['No. '])
        val = row['Final categories'].strip()
        cat_map[cat] = val
    return cat_map


def load_keyword_map(filename, include_other=False):
    #print 'loading keyword map...'
    keyword_map = {}
    with open(filename, 'r') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            kw = row['terms'].strip().lower()
            clust = row['category']
            if str(clust) == '45':
                clust = 43
            if str(clust) == '' or int(clust) > 43: # Check for empty category
                if include_other:
                    clust = 0
                else:
                    continue
            clust = int(clust)
            if kw not in keyword_map or keyword_map[kw] == '0':
                keyword_map[kw] = clust
    print(str(len(keyword_map.keys())), ' keywords loaded')
    return keyword_map


def map_keyword(keyword, cat_map):
    if keyword in cat_map:
        kw_cat = cat_map[keyword]
    else:
        kw_cat = '0'
    return kw_cat

# Word vector functions #########################

def get_w2v(word, model):
    dim = model.vector_size
    if word in model:
        return list(model[word])
    else:
        return zero_vec(dim)


def load_w2v(filename):
    if '.bin' in filename:
        model = load_bin_vectors(filename, True)
    elif '.wtv' in filename:
        model = Word2Vec.load(filename)
    else:
        model = load_bin_vectors(filename, False)
    dim = model.vector_size
    return model, dim


def load_bin_vectors(filename, bin_vecs=True):
    word_vectors = KeyedVectors.load_word2vec_format(filename, binary=bin_vecs, unicode_errors='ignore')
    return word_vectors


def zero_vec(dim):
    vec = []
    for x in range(dim):
        vec.append(0)
    return vec
