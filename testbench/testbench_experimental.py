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

# load and prepare data 
full_data = dte.load_data(PATHS_TO_DATA, PATH_TO_CATS)
prepared_data = dte.prepare_data(full_data)
# print(len(prepared_data))
# print(prepared_data[0:3])
shuffle(prepared_data)
print('prepared data size: {}'.format(str(len(prepared_data))))

print(full_data[-3:])
sys.exit()

# map class labels to indices

class_label_set = [data_combo[1] for data_combo in prepared_data]
base_labels = list(set(class_label_set))
max_label = max(base_labels)
unique_labels = list(range(0, max_label+1))


# create the embedding matrix
deva_index, embed_mat = dte.word_vectorize_data(prepared_data)
# print(embed_mat[:3,:])
print('EMBED MAT:')
print(embed_mat.shape)
# print(embed_mat[:3,:])


# ===== uncomment for training =====

# TRAIN TEST SPLITTING
PERCENT_TRAIN = 0.80
cutoff_index = int(np.floor(len(prepared_data) * PERCENT_TRAIN)) 
if cutoff_index == len(prepared_data):
    cutoff_index = cutoff_index - 1

train_data = prepared_data[:cutoff_index]
test_data = prepared_data[cutoff_index:]
print('train set size: {}'.format(str(len(train_data))))
print('test set size: {}'.format(str(len(test_data))))

# store the test data
# with open('./test_set_1.pk', 'wb') as dumpfile:
#     pk.dump(test_data, dumpfile)

param_dict_cnntext = {
    'batch_size': 16,
    'epoch_num': 3,
    'lr': 0.03,
    'kernel_num': 200,
    'kernel_sizes': 5,
    'hidden_size': 100,
    'dropout_rate': 0.0,
    'use_cuda': False,
    'unique_labels': unique_labels
}

param_dict_cnnrnn = {
    'batch_size': 16,
    'num_epochs': 3,
    'lr': 0.03,
    'kernel_num': 200,
    'kernel_sizes': 5,
    'hidden_size': 100,
    'dropout_rate': 0.0,
    'use_cuda': False,
    'unique_labels': unique_labels
}

param_dict_basernn = {
    'batch_size': 10,
    'num_epochs': 3,
    'lr': 0.001,
    # 'weight_decay': 0.2,
    'deva_index': deva_index,
    'embed_mat': embed_mat,
    'unique_labels': unique_labels
}

param_dict_seqcnn = {
    'batch_size': 16,
    'num_epochs': 5,
    'lr': 0.03,
    'deva_index': deva_index,
    'embed_mat': embed_mat,
    'unique_labels': unique_labels,
    'seq_len': 60
}

# CNN and CNN-RNN

# cnntext_y = np.array([data_combo[1] for data_combo in train_data])
# cnntext_x = dte.sentence_block_data(train_data, max_len_val=None)

# cnntext_model = trainer.train_model('cnntext', cnntext_x, cnntext_y, param_dict=param_dict_cnntext)
# trainer.save_model(cnntext_model, './cnntext_test_3.pt')

# cnn_model, rnn_model = trainer.train_model('cnnrnn', cnntext_x, cnntext_y, param_dict=param_dict_cnnrnn)
# trainer.save_model(cnn_model, './cnnrnn_test_3_cnn.pt')
# trainer.save_model(rnn_model, './cnnrnn_test_3_rnn.pt')

# Base-RNN and SeqCNN
# labels = np.array([data_combo[1] for data_combo in train_data]) # resize as (num_sentences, 1)
# max_seq_len = max([len(data_combo[0].split()) for data_combo in prepared_data])
# print('max:')
# print(max_seq_len)
# mean_seq_len = np.mean(np.array([len(data_combo[0].split()) for data_combo in prepared_data]))
# print('mean:')
# print(mean_seq_len)
# print('list of lengths over entire data:')
# length_list = [len(data_combo[0].split()) for data_combo in prepared_data]
# bin_count = np.bincount(np.array(length_list))
# ints = np.nonzero(bin_count)[0]
# print([x for x in zip(ints, bin_count[ints])])

SEQ_LEN = 60

# feature_sequences = [convert_token_to_index_sequence(deva_index, data_combo[0], SEQ_LEN) for data_combo in train_data]
# input_matrix = np.matrix(feature_sequences)

# basernn_y = labels
# basernn_x = input_matrix
# print(basernn_x.shape, basernn_y.shape)

seqcnn_y = np.array([data_combo[1] for data_combo in train_data])
seqcnn_x = dte.sentence_block_data(train_data, max_len_val=SEQ_LEN)
seqcnn_x = torch.squeeze(torch.FloatTensor(seqcnn_x), 1)
print(seqcnn_x.shape)

seqcnn_model = trainer.train_model('seqcnn2d', seqcnn_x, seqcnn_y, param_dict_seqcnn)
trainer.save_model(seqcnn_model, './seqcnn2d_test_1.pt')

# print('starting training ...')
# basernn_model, final_hidden = trainer.train_model('basernn', basernn_x, basernn_y, param_dict_basernn)
# trainer.save_model(basernn_model, './basernn_test_3.pt')
# print('training complete.')
# saved_hidden = final_hidden

# ===== uncomment for training =====


########################

'''
# CNN stuff

labels = np.array([data_combo[1] for data_combo in prepared_data]) # resize as (num_sentences, 1)

cnntext_data = dte.sentence_block_data(prepared_data)
print(cnntext_data.shape)
# print(len(class_label_set))

param_dict = {
    'batch_size': 16,
    'num_epochs': 3,
    'lr': 0.03,
    'kernel_num': 200,
    'kernel_sizes': 5,
    'hidden_size': 100,
    'dropout_rate': 0.0,
    'use_cuda': False
}


cnn_model = mdl.CNNText(300, len(unique_labels), dropout=0.0, ensemble=True)
rnn_model = mdl.TextRNNClassifier(param_dict['kernel_num'] * param_dict['kernel_sizes'], hidden_size = param_dict['hidden_size'], output_size = len(unique_labels))

cnn_opt = torch.optim.Adam(cnn_model.parameters(), lr=param_dict['lr'])
rnn_opt = torch.optim.Adam(rnn_model.parameters(), lr=param_dict['lr'])
# divide into batches
batch_size = param_dict['batch_size']
epoch_num = param_dict['num_epochs']
index_chunk_list = list(chunks(list(range(0, cnntext_data.shape[0])), batch_size))
# print(index_chunk_list)

# cuda, loss criterion, and optimizer
use_cuda = 0
criterion = nn.CrossEntropyLoss()

cnn_model.train()
rnn_model.train()
for i in range(epoch_num):
    print('current epoch: {}'.format(str(i)))
    for index_chunk in index_chunk_list:

        X_batch = cnntext_data[index_chunk, :, :, :]
        y_batch = labels[index_chunk]

        print(X_batch.shape, y_batch.shape)

        features = torch.FloatTensor(X_batch)
        target = torch.LongTensor(y_batch)

        # print(type(features), type(target))

        cnn_opt.zero_grad()

        cnn_out = cnn_model(features)
        # print(target.shape)
        # print(cnn_out.shape)

        print(cnn_out.shape)
        # checker = [j for j in range(batch_size) if z + j < cnntext_data.shape[0]]
        # print(checker)
        # pred_labels_tuple = torch.max(basernn_out, dim=1)
        # pred_labels = pred_labels_tuple[1] 
        # print(target)
        # print(pred_labels)
        # one at a time ...
        for j in range(cnn_out.shape[0]):
            rnn_out = rnn_model(cnn_out[j], 1)
            loss = F.cross_entropy(rnn_out, torch.argmax(target[j]).reshape((1,)))
        # loss = criterion(cnn_out, target)
        loss.backward()
        cnn_opt.step()
        rnn_opt.step()

'''
'''
RNN STUFF 
# construct the models
base_rnn = mdl.BaseRNN(len(list(deva_index.keys())), 300, embed_mat, len(unique_labels))
print('all good so far.')

# test a training of the base_rnn 
# ---
# turn data set into integer sequence (num_sentences, max_seq_length) and target sequence (num_sentences, 1)
labels = np.array([data_combo[1] for data_combo in prepared_data]) # resize as (num_sentences, 1)
max_seq_len = max([len(data_combo[0].split()) for data_combo in prepared_data])
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

base_rnn.train()
for i in range(epoch_num):
    print('current epoch: {}'.format(str(i)))
    for index_chunk in index_chunk_list:

        X_batch = input_matrix[index_chunk, :]
        y_batch = labels[index_chunk]

        print(X_batch.shape, y_batch.shape)

        features = torch.LongTensor(X_batch)
        target = torch.LongTensor(y_batch)

        # print(type(features), type(target))

        optimizer.zero_grad()

        basernn_out = base_rnn(features)
        print(target.shape)
        print(basernn_out.shape)
        # pred_labels_tuple = torch.max(basernn_out, dim=1)
        # pred_labels = pred_labels_tuple[1] 
        # print(target)
        # print(pred_labels)
        loss = criterion(basernn_out, target)
        loss.backward()
        optimizer.step()


# train and save the models 
'''

#####################################################################

# TESTING 

# LOAD TEST DATA

'''
test_data = None
with open('./test_set_1.pk', 'rb') as f:
    test_data = pk.load(f)

print(len(test_data))
'''

# repositioned param dicts

param_dict_cnntext = {
    'batch_size': 16,
    'epoch_num': 3,
    'lr': 0.03,
    'kernel_num': 200,
    'kernel_sizes': 5,
    'hidden_size': 100,
    'dropout_rate': 0.0,
    'use_cuda': False,
    'unique_labels': unique_labels
}

param_dict_cnnrnn = {
    'batch_size': 16,
    'num_epochs': 3,
    'lr': 0.03,
    'kernel_num': 200,
    'kernel_sizes': 5,
    'hidden_size': 100,
    'dropout_rate': 0.0,
    'use_cuda': False,
    'unique_labels': unique_labels
}

param_dict_basernn = {
    'batch_size': 10,
    'num_epochs': 3,
    'lr': 0.03,
    'weight_decay': 0.2,
    'deva_index': deva_index,
    'embed_mat': embed_mat,
    'unique_labels': unique_labels
}

param_dict_seqcnn = {
    'batch_size': 16,
    'num_epochs': 10,
    'lr': 0.03,
    'deva_index': deva_index,
    'embed_mat': embed_mat,
    'unique_labels': unique_labels,
    'seq_len': 60
}

# LOAD MODELS



# cnntext_model = mdl.CNNText(300, len(param_dict_cnntext['unique_labels']), dropout=param_dict_cnntext['dropout_rate'], ensemble=False)
# cnntext_model.load_state_dict(torch.load('./cnntext_test_3.pt'))
# cnntext_model.eval()

# cnnrnn_cnn_model = mdl.CNNText(300, len(param_dict_cnnrnn['unique_labels']), dropout=param_dict_cnnrnn['dropout_rate'], ensemble=True)
# cnnrnn_rnn_model = mdl.TextRNNClassifier(param_dict_cnnrnn['kernel_num'] * param_dict_cnnrnn['kernel_sizes'], hidden_size = param_dict_cnnrnn['hidden_size'], output_size = len(param_dict_cnnrnn['unique_labels']))
# cnnrnn_cnn_model.load_state_dict(torch.load('./cnnrnn_test_3_cnn.pt'))
# cnnrnn_rnn_model.load_state_dict(torch.load('./cnnrnn_test_3_rnn.pt'))
# cnnrnn_cnn_model.eval()
# cnnrnn_rnn_model.eval()


# rnn_base_model =  mdl.BaseRNN(len(list(param_dict_basernn['deva_index'].keys())), 300, param_dict_basernn['embed_mat'], len(param_dict_basernn['unique_labels']), hidden_size=12)
# rnn_base_model.load_state_dict(torch.load('./basernn_test_3.pt'))
# rnn_base_model.eval()

seqcnn_model = mdl.SequenceCNN2D(300, len(list(param_dict_seqcnn['deva_index'].keys())), len(param_dict_seqcnn['unique_labels']), param_dict_seqcnn['seq_len'])
seqcnn_model.load_state_dict(torch.load('./seqcnn2d_test_1.pt'))
seqcnn_model.eval()

print('All ready.')

# eval cnntext model

# cnntext_y_test = np.array([data_combo[1] for data_combo in test_data])
# cnntext_x_test = dte.sentence_block_data(test_data)

# print(cnntext_y_test.shape)
# print(cnntext_x_test.shape)

# eval basernn model 
# basernn_y_test = np.array([data_combo[1] for data_combo in test_data])
#### max_seq_len = max([len(data_combo[0].split()) for data_combo in prepared_data])
# feature_sequences = [convert_token_to_index_sequence(deva_index, data_combo[0], SEQ_LEN) for data_combo in test_data]
# input_matrix = np.matrix(feature_sequences)
# basernn_x_test = input_matrix
# print(basernn_x_test.shape)

# eval seqcnn model (old)

#seqcnn_y_test = np.array([data_combo[1] for data_combo in test_data])
#### max_seq_len = max([len(data_combo[0].split()) for data_combo in prepared_data])
# feature_sequences = [convert_token_to_index_sequence(deva_index, data_combo[0], SEQ_LEN) for data_combo in test_data]
# input_matrix = np.matrix(feature_sequences)
# seqcnn_x_test = input_matrix

# eval seqcnn model (new)

seqcnn_y_test = np.array([data_combo[1] for data_combo in test_data])
seqcnn_x_test = dte.sentence_block_data(test_data, max_len_val=SEQ_LEN)
seqcnn_x_test = torch.squeeze(torch.FloatTensor(seqcnn_x_test), 1)

total = 0
correct = 0

# seqcnn


categoricals = np.zeros(len(unique_labels)+1)
with torch.no_grad():
    print(seqcnn_x_test.shape)
    for i in range(seqcnn_x_test.shape[0]):
        input_data = seqcnn_x_test[i, :, :]
        input_data = torch.unsqueeze(input_data, 0)
        label = seqcnn_y_test[i]
        features = torch.FloatTensor(input_data)
        # print('ready ........')
        outputs = seqcnn_model(features)
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


# cnntext

'''
categoricals = np.zeros(len(unique_labels)+1)
with torch.no_grad():
    for i in range(cnntext_x_test.shape[0]):
        input_data = cnntext_x_test[i,:,:,:]
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
# print(basernn_x_test.shape, basernn_y_test.shape)


# cnn + rnn

'''
with torch.no_grad():
    for i in range(cnntext_x_test.shape[0]):
        input_data = cnntext_x_test[i,:,:,:]
        label = cnntext_y_test[i]
        features = torch.FloatTensor(input_data)
        feature_check = features.unsqueeze(0)
        outputs = cnnrnn_cnn_model(feature_check)
        rnn_out = cnnrnn_rnn_model(outputs[0])
        print(rnn_out)
        _, predicted = torch.max(rnn_out.data, 1)
        print(i, label, predicted)
        total += 1
        correct += (predicted == label)

print(total, correct.item(), (correct.item()/total))
'''

# base rnn

# rnn_base_model.hidden = final_hidden
'''
categoricals = np.zeros(unique_labels+1)
with torch.no_grad():
    for i in range(basernn_x_test.shape[0]):
        input_data = basernn_x_test[i,:]
        label = basernn_y_test[i]
        features = torch.LongTensor(input_data)
        # print('ready ........')
        output = rnn_base_model(features)
        print(output)
        _, predicted = torch.max(output.data, 1)
        print(i, label, predicted)
        categoricals[predicted.item()] += 1
        total += 1
        correct += (predicted == label)

print(total, correct.item(), (correct.item()/total))
print(categoricals)
print(basernn_x_test.shape, basernn_y_test.shape)
'''

