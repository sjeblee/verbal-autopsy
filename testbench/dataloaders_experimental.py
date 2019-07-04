# from transliteration_conversion import *
import xml.etree.ElementTree as ET 
import pprint 
from gensim.models.wrappers import FastText
import string 
import numpy as np 
import pandas as pd 

hi_model = FastText.load_fasttext_format('../wiki.hi.bin')

pp = pprint.PrettyPrinter(indent=4)

def load_data(data_path, category_path):
    '''
    input: 
        data_path => path to XML data 
    output: 
        [
            {'MG_ID': id num,
             'narrative': devanagari narrative,
             'original_narr': original transcribed,
             'Final_code': death classification,
             'cghr_cat': category integer}
        ]
    '''
    root = ET.parse(data_path).getroot()

    # load catmap
    cat_frame = pd.read_csv(category_path, header=None, names=['Final_code', 'cghr_cat'])
    cat_idx = cat_frame.set_index('Final_code')
    catmap = cat_idx.to_dict()['cghr_cat']

    full_data = []

    for AA in root.iter('Adult_Anonymous'):
        curr_adult = {}
        for m in AA.iter('MG_ID'):
            curr_adult['MG_ID'] = int(m.text)
        for n in AA.iter('narrative'):
            curr_adult['narrative'] = n.text 
        for o in AA.iter('original_narr'):
            curr_adult['original_narr'] = o.text 
        for p in AA.iter('Final_code'):
            curr_adult['Final_code'] = p.text
            curr_adult['cghr_cat'] = catmap[p.text] 
        full_data.append(curr_adult)


    return full_data  

def preprocess(data_string):
    '''
    What it does:
        - parses all punctuation from string
        - parses non-devanagari text from string
    '''
    p_str = data_string.translate(str.maketrans('','',string.punctuation))
    return p_str

def prepare_data(data_array):
    '''
    input: 
        [
            {'MG_ID': id num,
             'narrative': devanagari narrative,
             'original_narr': original transcribed,
             'Final_code': death classification,
             'cghr_cat': category integer}
        ]
    output:
        [
            [devanagari, cghr_cat]
        ]
    '''

    # since we have the devanagari here already, skip most preproc for now
    # just strip punctuation
    parsed_data = []
    for record in data_array:
        try:
            parsed_deva = preprocess(record['narrative'])
            if len(parsed_deva) != 0:
                parsed_data.append([parsed_deva, record['cghr_cat']])
        except KeyError:
            pass 

    return parsed_data 

def sentence_vectorize_data(data, ft_model):
    # tf idf score over all
    # N x 300 sentence vector collection with N x 1 categories 
    return

def word_vectorize_data(data):
    '''
    input:
        data => 
            [
                [devanagari, final_code]
            ]
        ft_model => fasttext model for hindi (loaded externally)
    '''
    # Task: Given the data format, output the devanagari to index dict, and the
    # embedding matrix
    
    deva_to_idx = {}
    idx_ft_vector_mat = [np.zeros(300)]

    full_deva_soup = []
    for deva_record in data:
        full_deva_soup += deva_record[0].split()
    
    unique_deva = list(set(full_deva_soup))
    len('unique devanagari: {}'.format(str(len(unique_deva))))

    i = 1 
    missed = 0 
    for word in unique_deva:
        deva_to_idx[word] = i
        try: 
            idx_ft_vector_mat.append(hi_model[word])
        except KeyError:
            idx_ft_vector_mat.append(np.zeros(300))
            missed += 1
        i += 1
    deva_to_idx['<PAD>'] = 0
    deva_to_idx['<UNK>'] = i
    idx_ft_vector_mat.append(np.full((300,), 999.9))
    idx_mat = np.matrix(idx_ft_vector_mat) 
    print('missed {} tokens'.format(str(missed)))
    
    return deva_to_idx, idx_mat

