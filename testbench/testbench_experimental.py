import dataloaders_experimental as dte 
import models_experimental as mdl 
import train_experimental as tr
import sys 

import numpy as np 

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

PATH_TO_DATA = './hindi_all_adult.xml'
PATH_TO_CATS = './cghr_code_map_adult.csv'

def convert_token_to_index_sequence(vocab_dict, data_string, max_seq_len):
    tokenized = data_string.split()
    integer_idx_list = [vocab_dict[token] for token in tokenized]
    padding_amount = max_seq_len - len(integer_idx_list)
    zero_padding = list(np.zeros(padding_amount, dtype=np.int))
    return integer_idx_list + zero_padding

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

# load and prepare data 
full_data = dte.load_data(PATH_TO_DATA, PATH_TO_CATS)
prepared_data = dte.prepare_data(full_data)
# print(len(prepared_data))

# map class labels to indices

class_label_set = [data_combo[1] for data_combo in prepared_data]
unique_labels = list(set(class_label_set))
# print(len(unique_labels))


# create the embedding matrix
deva_index, embed_mat = dte.word_vectorize_data(prepared_data)
# print(embed_mat[:3,:])

# construct the models
base_rnn = mdl.BaseRNN(len(list(deva_index.keys())), 300, embed_mat, len(unique_labels))
print('all good so far.')

# test a training of the base_rnn 
# ---
# turn data set into integer sequence (num_sentences, max_seq_length) and target sequence (num_sentences, 1)
labels = np.array([data_combo[1] for data_combo in prepared_data]) # resize as (num_sentences, 1)
max_seq_len = max([len(data_combo[0]) for data_combo in prepared_data])
feature_sequences = [convert_token_to_index_sequence(deva_index, data_combo[0], max_seq_len) for data_combo in prepared_data]
input_matrix = np.matrix(feature_sequences)

# -----
# shuffle 
# ( skip for now )

# divide into batches
batch_size = 16
epoch_num = 1
index_chunk_list = list(chunks(list(range(0, input_matrix.shape[0])), batch_size))
# print(index_chunk_list)

# cuda, loss criterion, and optimizer
use_cuda = 0
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_rnn.parameters()), lr=0.01, weight_decay=0.2)

# set cuda
if use_cuda:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    base_rnn.cuda()
    criterion = criterion.cuda()

print('all ready.')
# train 

'''
for i in range(epoch_num):
    for index_chunk in index_chunk_list:

        X_batch = input_matrix[index_chunk, :]
        y_batch = labels[index_chunk]

        print(X_batch.shape, y_batch.shape)

        features = nn.Variable(X_batch)
        target = nn.Variable(y_batch)

        optimizer.zero_grad()

        basernn_out = rnn(features)
        loss = criterion(basernn_out, target)
        loss.backward()
        optimizer.step()
'''



# train and save the models 