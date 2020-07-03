import numpy as np
import dataloaders_experimental as dte 
from random import shuffle 
from keras.datasets import imdb
from keras import optimizers 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
from keras.utils import to_categorical
np.random.seed(7)

	
# load the dataset but only keep the top n words, zero the rest
# top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)


# truncate and pad input sequences
# max_review_length = 500
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

# create the hindi fasttext embedding 

PATH_TO_DATA = './hindi_all_adult.xml'
PATH_TO_CATS = './cghr_code_map_adult.csv'

PATHS_TO_DATA = ['./hindi_all_adult.xml', './all_adult_cat_hi.xml']

def convert_token_to_index_sequence(vocab_dict, data_string, seq_len=60):
    tokenized = data_string.split()
    integer_idx_list = []
    for token in tokenized:
    	try:
    		integer_idx_list.append(vocab_dict[token])
    	except:
    		integer_idx_list.append(vocab_dict['<UNK>'])
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

max_length = 60

# create integer sequences 
integerized_data = []
for data_set in prepared_data:
	x_string = data_set[0]
	y_t = data_set[1]
	integer_set = convert_token_to_index_sequence(deva_index, x_string, max_length)
	integerized_data.append([integer_set, y_t])

# ===== uncomment for training =====

# TRAIN TEST SPLITTING
PERCENT_TRAIN = 0.80
cutoff_index = int(np.floor(len(prepared_data) * PERCENT_TRAIN)) 
if cutoff_index == len(prepared_data):
    cutoff_index = cutoff_index - 1

train_data = integerized_data[:cutoff_index]
test_data = integerized_data[cutoff_index:]
print('train set size: {}'.format(str(len(train_data))))
print('test set size: {}'.format(str(len(test_data))))

X_train = np.matrix([np.array(data_combo[0]) for data_combo in train_data])
y_train = np.array([data_combo[1] for data_combo in train_data])
y_train_cat = to_categorical(y_train)

X_test = np.matrix([np.array(data_combo[0]) for data_combo in test_data])
y_test = np.array([data_combo[1] for data_combo in test_data])
y_test_cat = to_categorical(y_test)

# create the model
embedding_vector_length = 300
n_words = embed_mat.shape[0]
model = Sequential()
# model.add(Embedding(n_words, embedding_vector_length, weights=[embed_mat], input_length=max_length, trainable=False)) # replace 
model.add(LSTM(60))
model.add(Dense(19, activation='softmax'))
sgd = optimizers.SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train_cat, validation_data=(X_test, y_test_cat), epochs=20, batch_size=16)

# Final evaluation of the model 
outputs = model.predict(X_test, verbose=1)
print(outputs)
print(y_test_cat)

# print("Accuracy: %.2f%%" % (scores[1]*100))

