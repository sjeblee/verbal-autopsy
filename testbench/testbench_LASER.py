import dataloaders_experimental as dte 
import models_experimental as mdl 
import train_experimental as trainer
import sys 

import numpy as np 
from random import shuffle 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

import pickle as pk

PATH_TO_DATA = './hindi_all_adult.xml'
PATH_TO_CATS = './cghr_code_map_adult.csv'

PATHS_TO_DATA = ['./hindi_all_adult.xml', './all_adult_cat_hi.xml']

# ---

def convert_sentences_to_embedding_block(sentence_dict, embed_matrix, data_list, seq_len, embed_size=1024):
	'''
	in: 
		data list -> [[hin/eng sentences list 1], [hin/eng sentences list 2], ...]

	out: (N, seq_len, 1024)
	'''

	missed = 0
	total = 0

	full_block = []

	for sentences in data_list:
		# sentences is a list
		part_chunk = []
		final = max(0, (len(sentences)-1) - seq_len)
		for i in range(len(sentences)-1, final, -1):
			try:
				sent = sentences[i]
				embedding_index = sentence_dict[sent]
				part_chunk.append(embed_matrix[embedding_index, :])
			except:
				missed += 1
				part_chunk.append(np.zeros(embed_size))
			total += 1

		if len(part_chunk) < seq_len:

			remaining = seq_len - len(part_chunk)
			part_chunk += [np.zeros(embed_size) for x in range(remaining)]

		assert len(part_chunk) == seq_len
		full_block.append(part_chunk)

	print('missed: ')
	print(missed)
	print('total: ')
	print(total)
	print('ratio:')
	print(str(float(missed / total)))

	X_matrix = np.array(full_block)
	print('dimensions:')
	print(X_matrix.shape)

	return X_matrix

def convert_token_to_index_sequence(vocab_dict, data_string, seq_len=60):
    tokenized = data_string.split()
    integer_idx_list = [vocab_dict[token] for token in tokenized]
    sentence_length = len(integer_idx_list)
    full_list = []
    if sentence_length < seq_len:
        padding_amount = seq_len - sentence_length
        zero_padding = list(np.zeros(padding_amount, dtype=np.int))
        full_list = integer_idx_list + zero_padding
    else:
        remove_token_number = sentence_length - seq_len 
        full_list = integer_idx_list[remove_token_number:]
    return full_list

'''
def convert_token_to_index_sequence(vocab_dict, data_string, max_seq_len):
    tokenized = data_string.split()
    integer_idx_list = [vocab_dict[token] for token in tokenized]
    padding_amount = max_seq_len - len(integer_idx_list)
    zero_padding = list(np.zeros(padding_amount, dtype=np.int))
    return integer_idx_list + zero_padding
'''

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# ---

full_data = dte.load_data(PATHS_TO_DATA, PATH_TO_CATS)
eng_narr_data = [x for x in full_data if x['eng_narrative'] is not None] # this will be full data to split

# --- 

eng_data = [x['eng_narrative'] for x in eng_narr_data]
hin_data = [x['narrative'] for x in eng_narr_data]

# print(len(eng_data), len(hin_data))

eng_soup, hin_soup = dte.create_sentence_soup(eng_data, hin_data)
# eng_idx_dict, hin_idx_dict = dte.create_sentence_dicts(eng_soup, hin_soup)


# --- 

# create the files and save the dictionaries 

'''
dte.create_sentence_files(eng_soup, hin_soup) 

dict_file_eng = './eng_idx_dict.pk'
dict_file_hin = './hin_idx_dict.pk'

with open(dict_file_eng, 'wb') as f:
	pk.dump(eng_idx_dict, f)
print('english dumped')

with open(dict_file_hin, 'wb') as g:
	pk.dump(hin_idx_dict, g)
print('hindi dumped')
'''

# ---

# unique labels

class_label_set = [x['cghr_cat'] for x in eng_narr_data]
base_labels = list(set(class_label_set))
max_label = max(base_labels)
unique_labels = list(range(0, max_label+1))

# parse data

parsed_data = dte.parse_laser_data(eng_narr_data)
# x[0] -> eng 
# x[1] -> hin
# x[2] -> cghr cat 

print('parsed data len')
print(len(parsed_data))

# max sequence length

max_len_en = np.ceil(np.mean(np.array([len(x[0]) for x in parsed_data])))
max_len_hi = np.ceil(np.mean(np.array([len(x[1]) for x in parsed_data])))

max_len = int(max_len_en)
if max_len_hi > max_len:
	max_len = int(max_len_hi) 

print('max len:')
print(max_len)

# shuffle data 

shuffle(parsed_data)

# data split

PERCENT_TRAIN = 0.80
cutoff_index = int(np.floor(len(parsed_data) * PERCENT_TRAIN)) 
if cutoff_index == len(parsed_data):
    cutoff_index = cutoff_index - 1

train_data = parsed_data[:cutoff_index]
test_data = parsed_data[cutoff_index:]
print('train set size: {}'.format(str(len(train_data))))
print('test set size: {}'.format(str(len(test_data))))

dim = 1024

# --- TRAINING 

# load eng embeddings and dict

eng_embed = np.fromfile('./eng_embeddings.raw', dtype=np.float32, count=-1)
eng_embed.resize(eng_embed.shape[0] // dim, dim)
print('eng embed dimensions:')
print(eng_embed.shape)

eng_dict = pk.load(open('./eng_idx_dict.pk', 'rb'))
print('eng dict len:')
print(len(eng_dict))

# extract out targets 

y_train = np.array([x[2] for x in train_data])
x_train_list_eng = [x[0] for x in train_data]

print('x len and y len:')
print(len(x_train_list_eng), len(y_train))

# convert sentences to data block 

X_train = convert_sentences_to_embedding_block(eng_dict, eng_embed, x_train_list_eng, max_len)
print(X_train.shape)

# print(X_train[0:3,:,:])

# --- TESTING

hin_embed = np.fromfile('./hin_embeddings.raw', dtype=np.float32, count=-1)
hin_embed.resize(hin_embed.shape[0] // dim, dim)
print('hin embed dimensions:')
print(hin_embed.shape)

hin_dict = pk.load(open('./hin_idx_dict.pk', 'rb'))
print('hin dict len:')
print(len(hin_dict))

# extract out targets 

y_test = np.array([x[2] for x in test_data])
x_test_list_hin = [x[1] for x in test_data]

print('x len and y len:')
print(len(x_test_list_hin), len(y_test))

# convert sentences to data block 

X_test = convert_sentences_to_embedding_block(hin_dict, hin_embed, x_test_list_hin, max_len)
print(X_test.shape)

# --- MODEL 

# CNNText 

'''
param_dict_cnntext = {
    'batch_size': 16,
    'epoch_num': 3,
    'lr': 0.03,
    'kernel_num': 200,
    'kernel_sizes': 5,
    'hidden_size': 100,
    'dropout_rate': 0.0,
    'use_cuda': False,
    'unique_labels': unique_labels,
    'embed_size': 1024
}

cnntext_x = X_train 
cnntext_y = y_train

cnntext_model = trainer.train_model('cnntext', cnntext_x, cnntext_y, param_dict=param_dict_cnntext)
trainer.save_model(cnntext_model, './cnntext_test_LASER_1.pt')
'''


# Seq2DCNN 

'''
# A - 3456
# B - 5678


param_dict_seqcnn2d = {
    'batch_size': 16,
    'num_epochs': 5,
    'lr': 0.03,
    'deva_index': eng_dict,
    'embed_size': dim,
    'unique_labels': unique_labels,
    'seq_len': max_len,
    'conv_filters': 64
}

cnntext_x = X_train 
cnntext_y = y_train

cnntext_model = trainer.train_model('seqcnn2d', cnntext_x, cnntext_y, param_dict=param_dict_seqcnn2d)
trainer.save_model(cnntext_model, './seqcnn2d_test_LASER_1.pt')
'''

# Seq1DCNN

param_dict_seqcnn1d = {
    'batch_size': 16,
    'num_epochs': 5,
    'lr': 0.03,
    'deva_index': eng_dict,
    'embed_size': dim,
    'unique_labels': unique_labels,
    'seq_len': max_len,
    'conv_filters': 64
}

cnntext_x = X_train 
cnntext_y = y_train

cnntext_model = trainer.train_model('seqcnn', cnntext_x, cnntext_y, param_dict=param_dict_seqcnn1d)
trainer.save_model(cnntext_model, './seqcnn1d_test_LASER_1.pt')


# --- EVALUATION

# CNNText 

'''
cnntext_x_test = X_test 
cnntext_y_test = y_test

cnntext_model = mdl.CNNText(1024, len(param_dict_cnntext['unique_labels']), dropout=param_dict_cnntext['dropout_rate'], ensemble=False)
cnntext_model.load_state_dict(torch.load('./cnntext_test_LASER_1.pt'))
cnntext_model.eval()

total = 0
correct = 0

categoricals = np.zeros(len(unique_labels)+1)
with torch.no_grad():
    for i in range(cnntext_x_test.shape[0]):
        input_data = cnntext_x_test[i,:,:]
        label = cnntext_y_test[i]
        features = torch.FloatTensor(input_data)
        feature_check = features.unsqueeze(0)
        print(feature_check.size())
        outputs = cnntext_model(feature_check)
        _, predicted = torch.max(outputs.data, 1)
        print(i, label, predicted, int(predicted == label))
        categoricals[predicted.item()] += 1
        total += 1
        if (predicted == label):
            print('Good.')
            correct += 1
            print(correct)
        # correct += (predicted == label)

print(total, correct, (correct/total))
print(categoricals)
'''

# SeqCNN2D

'''
seqcnn_x_test = X_test 
seqcnn_y_test = y_test

seqcnn_model = mdl.SequenceCNN2D(1024, len(list(param_dict_seqcnn2d['deva_index'].keys())), len(param_dict_seqcnn2d['unique_labels']), param_dict_seqcnn2d['seq_len'], conv_filters=param_dict_seqcnn2d['conv_filters'])
seqcnn_model.load_state_dict(torch.load('./seqcnn2d_test_LASER_1.pt'))
seqcnn_model.eval()

total = 0
correct = 0

categoricals = np.zeros(len(unique_labels)+1)
with torch.no_grad():
    print(seqcnn_x_test.shape)
    for i in range(seqcnn_x_test.shape[0]):
        input_data = seqcnn_x_test[i, :, :]
        label = seqcnn_y_test[i]
        features = torch.FloatTensor(input_data)
        feature_check = torch.unsqueeze(features, 0)
        # print('ready ........')
        outputs = seqcnn_model(feature_check)
        _, predicted = torch.max(outputs.data, 1)
        print(i, label, predicted, int(predicted == label))
        categoricals[predicted.item()] += 1
        total += 1
        if (predicted == label):
            print('Good.')
            correct += 1
            print(correct)


print(total, correct, (correct/total))
print(categoricals)
'''

# SeqCNN1D

seqcnn_x_test = X_test 
seqcnn_y_test = y_test

seqcnn_model = mdl.SequenceCNN(1024, len(list(param_dict_seqcnn1d['deva_index'].keys())), len(param_dict_seqcnn1d['unique_labels']), param_dict_seqcnn1d['seq_len'], conv_filters=param_dict_seqcnn1d['conv_filters'])
seqcnn_model.load_state_dict(torch.load('./seqcnn1d_test_LASER_1.pt'))
seqcnn_model.eval()

total = 0
correct = 0

categoricals = np.zeros(len(unique_labels)+1)
with torch.no_grad():
    print(seqcnn_x_test.shape)
    for i in range(seqcnn_x_test.shape[0]):
        input_data = seqcnn_x_test[i, :, :]
        label = seqcnn_y_test[i]
        features = torch.FloatTensor(input_data)
        feature_check = torch.unsqueeze(features, 0)
        # print('ready ........')
        outputs = seqcnn_model(feature_check)
        _, predicted = torch.max(outputs.data, 1)
        print(i, label, predicted, int(predicted == label))
        categoricals[predicted.item()] += 1
        total += 1
        if (predicted == label):
            print('Good.')
            correct += 1
            print(correct)


print(total, correct, (correct/total))
print(categoricals)
















