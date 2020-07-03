# from transliteration_conversion import *
import xml.etree.ElementTree as ET 
import pprint 
from gensim.models.wrappers import FastText
import string 
import numpy as np 
import pandas as pd 

hi_model = FastText.load_fasttext_format('../wiki.hi.bin')

pp = pprint.PrettyPrinter(indent=4)

def load_data(data_paths, category_path):
    '''
    input: 
        data_paths => [paths to XML data] (in list)
    output: 
        [
            {'MG_ID': id num,
             'narrative': devanagari narrative,
             'original_narr': original transcribed,
             'Final_code': death classification,
             'cghr_cat': category integer}
        ]
    '''
    full_data_across_paths = []
    cat_frame = pd.read_csv(category_path, header=None, names=['Final_code', 'cghr_cat'])
    cat_idx = cat_frame.set_index('Final_code')
    catmap = cat_idx.to_dict()['cghr_cat']

    path_num = 0
    for path in data_paths:
        root = ET.parse(path).getroot()
    
        full_data = []

        for AA in root.iter('Adult_Anonymous'):
            curr_adult = {}
            for m in AA.iter('MG_ID'):
                curr_adult['MG_ID'] = str(m.text)
            if path_num == 0:
                for n in AA.iter('narrative'):
                    curr_adult['narrative'] = n.text
                curr_adult['eng_narrative'] = None
                for o in AA.iter('original_narr'):
                    curr_adult['original_narr'] = o.text
            else:
                curr_adult['eng_narrative'] = None
                for x in AA.iter('hindi_narrative'):
                    curr_adult['narrative'] = x.text
                for z in AA.iter('narrative'):
                    curr_adult['eng_narrative'] = z.text
                curr_adult['original_narr'] = None
            for p in AA.iter('Final_code'):
                curr_adult['Final_code'] = p.text
                curr_adult['cghr_cat'] = catmap[p.text] 
            full_data.append(curr_adult)

        path_num += 1
        full_data_across_paths += full_data

    return full_data

def parse_laser_data(eng_narr_data):
    '''in: the dict
    out: [[eng sent], [hin sent], cat]
    '''
    parsed_data = []
    for x in eng_narr_data:
        broken_up_en = [sent.strip().strip('\n').lower() for sent in x['eng_narrative'].split('.') if sent != '']
        broken_up_hi = [sent.strip().strip('\n') for sent in x['narrative'].split("।") if sent != '']
        data_point = [broken_up_en, broken_up_hi, x['cghr_cat']]
        parsed_data.append(data_point)

    return parsed_data

def create_sentence_soup(eng_data, hin_data):
    eng_sent_soup = []
    for a in eng_data:
        broken_up_en = [sent.strip().strip('\n').lower() for sent in a.split('.') if sent != '']
        eng_sent_soup += broken_up_en


    hin_sent_soup = []
    for b in hin_data:
        broken_up_hi = [sent.strip().strip('\n') for sent in b.split("।") if sent != '']
        hin_sent_soup += broken_up_hi

    print(len(eng_sent_soup), len(hin_sent_soup))

    eng_soup_unparsed = list(set(eng_sent_soup)) 
    hin_soup_unparsed = list(set(hin_sent_soup))

    print('uniques:')

    print(len(eng_soup_unparsed), len(hin_soup_unparsed))

    print('remove empties:')

    eng_soup = [x for x in eng_soup_unparsed if x != '' or x != ' ' or x != '\n']
    hin_soup = [x for x in hin_soup_unparsed if x != '' or x != ' ' or x != '\n']

    print(len(eng_soup), len(hin_soup))

    return eng_soup, hin_soup 

def create_sentence_dicts(eng_soup, hin_soup):
    eng_counter = 0
    eng_dict = {}
    for sent in eng_soup:
        eng_dict[sent] = eng_counter 
        eng_counter += 1

    hin_counter = 0
    hin_dict = {}
    for sent in hin_soup:
        hin_dict[sent] = hin_counter 
        hin_counter += 1

    return eng_dict, hin_dict 

def create_sentence_files(eng_soup, hin_soup):

    eng_fname = './eng_sentences.txt'
    hin_fname = './hin_sentences.txt'

    with open(eng_fname, 'w', encoding='utf-8') as f:
        for sent in eng_soup:
            f.write(sent)
            f.write('\n')

    print('Eng file written.')

    with open(hin_fname, 'w', encoding='utf-8') as g:
        for sent in hin_soup:
            g.write(sent)
            g.write('\n')

    print('Hin file written.')

    return

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

def sentence_block_data(data, max_len_val, embed_dim=300):
    '''
    input:
        data =>
            [
                [devanagari, final_code] 
            ]
        ft_model => fasttext model for hindi

    output:
        (N, 1, MAX_LEN, EMBED_DIM)
    '''

    N = len(data) # num setences
    if max_len_val == None:
        max_len = max([len(data_tuple[0].split()) for data_tuple in data])
    else:
        max_len = max_len_val

    X = np.zeros((N, 1, max_len, embed_dim))

    for i in range(N):
        tokenized_sentence = data[i][0].split()
        padding_amount = max_len - len(tokenized_sentence)
        embed_set = []
        max_len_counter = 0
        for token in tokenized_sentence:
            if max_len_counter < max_len:
                try:
                    embed_set.append(hi_model[token])
                except KeyError:
                    embed_set.append(np.zeros(embed_dim, dtype=np.int))
            else:
                break
            max_len_counter += 1
        if padding_amount >= 0:
            embed_set = embed_set + [np.zeros(embed_dim, dtype=np.int) for x in range(padding_amount)]
        X[i,0,:,:] = np.matrix(embed_set)

    return X 

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
    idx_ft_vector_mat.append(np.full((300,), 0))
    idx_mat = np.matrix(idx_ft_vector_mat) 
    # normalize 
    mean_vector = np.mean(idx_mat, axis=1)
    idx_mat_demean = idx_mat - mean_vector 
    total_values = idx_mat_demean.shape[0] - 2 # -2 to remove the <PAD> and <UNK> since they arent relevant
    normalized_idx_mat = idx_mat_demean / total_values
    print('missed {} tokens'.format(str(missed)))
    
    return deva_to_idx, normalized_idx_mat

