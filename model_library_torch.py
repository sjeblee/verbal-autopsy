#!/usr/bin/python3
# Neural network model functions in PyTorch
# @author sjeblee@cs.toronto.edu

import math
import numpy
import os
import random
import time
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker

import data_util3
import pickle

numpy.set_printoptions(threshold=numpy.inf)
#use_cuda = torch.cuda.is_available()
use_cuda = False
#torch.cuda.set_device(2)

# Set threshold for ill-defined class
threshold = 0.01

class CNN_Text(nn.Module):

     def __init__(self, embed_dim, class_num, kernel_num=200, kernel_sizes='12345', dropout=0.1):
          super(CNN_Text, self).__init__()
          #self.args = args

          #V = args.embed_num
          D = embed_dim
          C = class_num
          Ci = 1
          Co = kernel_num
          Ks = kernel_sizes

          self.conv11 = nn.Conv2d(Ci, Co, (1, D))
          self.conv12 = nn.Conv2d(Ci, Co, (2, D))
          self.conv13 = nn.Conv2d(Ci, Co, (3, D))
          self.conv14 = nn.Conv2d(Ci, Co, (4, D))
	  self.conv15 = nn.Conv2d(Ci, Co, (5, D))
	  #self.conv16 = nn.Conv2d(Ci, Co, (6, D))
	  #self.conv17 = nn.Conv2d(Ci, Co, (7, D))
	  #self.conv18 = nn.Conv2d(Ci, Co, (8, D))

          self.dropout = nn.Dropout(dropout)
          self.fc1 = nn.Linear(len(Ks)*Co, C)

     def conv_and_pool(self, x, conv):
          x = F.relu(conv(x)).squeeze(3)  # (N, Co, W)
          x = F.max_pool1d(x, x.size(2)).squeeze(2)
          return x

     def forward(self, x):
          #x = self.embed(x)  # (N, W, D)

          x = x.unsqueeze(1)  # (N, Ci, W, D)
          #x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]  # [(N, Co, W), ...]*len(Ks)
          #x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
          #x = torch.cat(x, 1)

          x1 = self.conv_and_pool(x,self.conv11) #(N,Co)
          x2 = self.conv_and_pool(x,self.conv12) #(N,Co)
          x3 = self.conv_and_pool(x,self.conv13) #(N,Co)
          x4 = self.conv_and_pool(x,self.conv14) #(N,Co)
	  x5 = self.conv_and_pool(x,self.conv15)
	  #x6 = self.conv_and_pool(x,self.conv16)
	  #x7 = self.conv_and_pool(x,self.conv17)
	  #x8 = self.conv_and_pool(x,self.conv18)
          x = torch.cat((x1, x2, x3, x4, x5), 1) # (N,len(Ks)*Co)

          #x = self.dropout(x)  # (N, len(Ks)*Co)
          logit = self.fc1(x)  # (N, C)
          return logit

# Neural Network Model (1 hidden layer)
class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
	#self.sm = nn.Softmax()

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
	#out = self.sm(out)
        return out

# Models from PyTorch Tutorial:
######################################################################
# The Encoder
# -----------
#
# The encoder of a seq2seq network is a RNN that outputs some value for
# every word from the input sentence. For every input word the encoder
# outputs a vector and a hidden state, and uses the hidden state for the
# next input word.
#
# .. figure:: /_static/img/seq-seq-images/encoder-network.png
#    :alt:
#
#

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #print "embedding init size: " + str(input_size)
        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(input_size, hidden_size)

    def forward(self, input, hidden):
        #print "forward: input: " + str(input.size())
        #embedded1 = self.embedding(input)
        embedded1 = input
        #print "embedded1: " + str(embedded1.size())
        embedded = embedded1.view(1, 1, -1)
        #print "embedded: " + str(embedded.size())
        #output = input
        output, hidden = self.gru(embedded, hidden)
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result

######################################################################
# The Decoder
# -----------
#
# The decoder is another RNN that takes the encoder output vector(s) and
# outputs a sequence of words to create the translation.
#
######################################################################
# Simple Decoder
# ^^^^^^^^^^^^^^
#
# In the simplest seq2seq decoder we use only last output of the encoder.
# This last output is sometimes called the *context vector* as it encodes
# context from the entire sequence. This context vector is used as the
# initial hidden state of the decoder.
#
# At every step of decoding, the decoder is given an input token and
# hidden state. The initial input token is the start-of-string ``<SOS>``
# token, and the first hidden state is the context vector (the encoder's
# last hidden state).
#
# .. figure:: /_static/img/seq-seq-images/decoder-network.png
#    :alt:
#
#

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        #print "forward: input: " + type(input)
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result
######################################################################
# Attention Decoder
# ^^^^^^^^^^^^^^^^^
#
# If only the context vector is passed betweeen the encoder and decoder,
# that single vector carries the burden of encoding the entire sentence.
#
# Attention allows the decoder network to "focus" on a different part of
# the encoder's outputs for every step of the decoder's own outputs. First
# we calculate a set of *attention weights*. These will be multiplied by
# the encoder output vectors to create a weighted combination. The result
# (called ``attn_applied`` in the code) should contain information about
# that specific part of the input sequence, and thus help the decoder
# choose the right output words.
#
# .. figure:: https://i.imgur.com/1152PYf.png
#    :alt:
#
# Calculating the attention weights is done with another feed-forward
# layer ``attn``, using the decoder's input and hidden state as inputs.
# Because there are sentences of all sizes in the training data, to
# actually create and train this layer we have to choose a maximum
# sentence length (input length, for encoder outputs) that it can apply
# to. Sentences of the maximum length will use all the attention weights,
# while shorter sentences will only use the first few.
#
# .. figure:: /_static/img/seq-seq-images/attention-decoder-network.png
#    :alt:
#
#

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=50):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        #self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        #print "encoder_outputs: " + str(encoder_outputs.size())
        #print "adecoder input: " + str(input.size())
        #embedded = self.embedding(input).view(1, 1, -1)
        embedded = input.view(1, 1, -1)
        #print "embedded: " + str(embedded.size())
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        #print "attn_weights: " + str(attn_weights.size())
        attn_applied = torch.bmm(attn_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        result = Variable(torch.zeros(1, 1, self.hidden_size))
        if use_cuda:
            return result.cuda()
        else:
            return result


######################################################################
# .. note:: There are other forms of attention that work around the length
#   limitation by using a relative position approach. Read about "local
#   attention" in `Effective Approaches to Attention-based Neural Machine
#   Translation <https://arxiv.org/abs/1508.04025>`__.

''' Create and train a feed-forward neural network model. Features selection should be done before passing to this function
    X: a python list or numpy array of training data of shape [num_samples, num_features]
    Y: a python list or numpy array of training labels of shape [num_samples]
    returns: the model and the modified X and Y arrays
'''
'''
def nn_model(X, Y, num_nodes, act, num_epochs=10):
    print("use_cuda", str(use_cuda))
    X = torch.Tensor(X)
    Y = torch.Tensor(Y).long()
    if use_cuda:
        X = X.cuda()
        Y = Y.cuda()
    print("X: ", str(X.size()))
    print("Y: ", str(Y.size()))

    # Hyper Parameters () # Print out the whole thing:
    input_dim = X.size()[-1] #100
    num_examples = X.size()[0] #11000
    num_classes = Y.size()[-1] #18

    batch_size = 100
    learning_rate = 0.001

    print "Input dimension: " + str(input_dim)
    print "Number of examples: " + str(num_examples)
    print "Number of classes: " + str(num_classes)

    print("neural network: nodes: ", str(num_nodes))
    net = Net(input_dim, num_nodes, num_classes)

    if use_cuda:
        net = net.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    # Train the Model
    net.train()
    for epoch in range(num_epochs):
        print("epoch", str(epoch))
        i = 0
        while (i+batch_size) < num_examples:
            if i%100000 == 0:
                print("batch i=", str(i))
            # Convert torch tensor to Variable
            samples = Variable(X[i:i+batch_size])
            labels = Variable(Y[i:i+batch_size])
	    print "Labels: " + str(labels)

            # Forward + Backward + Optimize
            optimizer.zero_grad()  # zero the gradient buffer
            outputs = net(samples)
            #loss = criterion(outputs, labels)
	    loss = criterion(outputs, torch.max(labels,1)[1])
	    print "Labels: " + str(torch.max(labels,1)[1])
            loss.backward()
            optimizer.step()

            if (i+1) % 10 == 0:
                print ('Epoch [%d/%d], Step [%d/%d], Loss: %.4f' %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))
            i = i+batch_size

    del X
    del Y
    if use_cuda:
        torch.cuda.empty_cache()
    return net
'''
def nn_model(X, Y, num_nodes, act, num_epochs=10):
	st = time.time()
	Xarray = numpy.asarray(X).astype('float')
	Yarray = Y.astype('int')
	print "X numpy shape: " + str(Xarray.shape) + "Y numpy shape: " + str(Yarray.shape)

	X_len = Xarray.shape[0]
	dim = Xarray.shape[-1]
	num_labels = Yarray.shape[-1]
	num_epochs = 10
	steps = 0
	best_acc = 0
	last_step = 0
	log_interval = 1000
	batch_size = 100
	num_batches = math.ceil(X_len/batch_size)
	learning_rate = 0.001

	net = Net(dim, num_nodes, num_labels)
	if use_cuda:
		net = net.cuda()

	# Train
	optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

	steps = 0
	net.train()
	for epoch in range(num_epochs):
		print "epoch: " + str(epoch)
		i = 0
		numpy.random.seed(seed=1)
		permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
		Xiter = Xarray[permutation]
		Yiter = Yarray[permutation]
	
		while i+batch_size < X_len:
			batchX = Xiter[i:i+batch_size]
			batchY = Yiter[i:i+batch_size]
			Xtensor = torch.from_numpy(batchX).float()
			Ytensor = torch.from_numpy(batchY).long()
			if use_cuda:
				Xtensor = Xtensor.cuda()
				Ytensor = Ytensor.cuda()
			feature = Variable(Xtensor)
			target = Variable(Ytensor)
			i = i+batch_size

			optimizer.zero_grad()
			logit = net(feature)

			loss = F.cross_entropy(logit, torch.max(target,1)[1])
			loss.backward()
			optimizer.step()

			steps+=1
		ct = time.time() - st
		unit = "s" 
		if ct> 60:
			ct = ct/60
			unit = "m"
		print "Time so far: " + str(ct) + unit
	return net


def test_nn(net, testX):
    # Test the Model

    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)
    testX = testX.astype('float')
    pred = []
    new_pred = [] # New prediction by assigning to UNKNOWN class if below the threshold
    probs = [] # Probabilities, the softmax output of the final layer

    batch_size = 100
    i = 0
    length = len(testX) #testX.shape[0]
    while i < length:
	pred = []
	probs = []
        if i%100000 == 0:
            print("test batch", str(i))
        if (i+batch_size) > length:
            batch_size = length-i
        if use_cuda:
            sample_tensor = torch.cuda.FloatTensor(testX[i:i+batch_size])
        else:
            sample_tensor = torch.FloatTensor(testX[i:i+batch_size])
        samples = Variable(sample_tensor)
        outputs = net(samples)

	outputs_ls = logsoftmax(outputs)
	outputs_softmax = softmax(outputs)
        #_, predicted = torch.max(outputs.data, 1)
	#probabilities, predicted = torch.max(outputs.data, 1)
	predicted = torch.max(outputs_ls, 1)[1]
	probabilities = torch.max(outputs_softmax, 1)[0]
        pred = pred + predicted.tolist()
	
	# Get the probabilities, and making new prediction based on random threshold
	probs = probs + probabilities.tolist()
	print "Predicted Lables: " + str(pred)
	print "Max Probabilties: " + str(probs)
	print "Not log probabilities: " + str(outputs)
	print "Probabilities: " + str(outputs_softmax)
	for num, prob in enumerate(probs):
	    if prob < threshold:  # Set the threshold. Initially hard-coded. Will be modified.
		new_pred.append(9) # What is the index location of UNKNOWN class? Figure it out!
	    else:
		new_pred.append(pred[num])

        del sample_tensor
        i = i+batch_size
    #return pred
    return new_pred

''' Creates and trains a recurrent neural network model. Supports SimpleRNN, LSTM, and GRU
    X: a list of training data
    Y: a list of training labels
'''
def rnn_model(X, Y, num_nodes, activation='sigmoid', modelname='lstm', dropout=0.1, X2=[], pretrainX=[], pretrainY=[], pretrainX2=[], initial_states=None, num_epochs=15):
    print("model: ", modelname, " nodes: ", str(num_nodes), " embedding: ", str(embedding_size), " max_seq_len: ", str(max_seq_len))

    n_iters = len(X)
    print("epochs: ", str(num_epochs), " iters: ", str(n_iters))
    if use_cuda:
        print("using cuda")
        X = torch.cuda.FloatTensor(X)
        Y = torch.cuda.FloatTensor(Y)
    else:
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    print("X: ", str(X.size()))
    print("Y: ", str(Y.size()))
    hidden_size = num_nodes
    input_dim = X.size()[-1]
    max_length = Y.size()[1]
    output_dim = Y.size()[-1]
    print("input_dim: ", str(input_dim))
    print("max_length: ", str(max_length))
    print("output_dim: ", str(output_dim))
    rnn = AttnDecoderRNN(hidden_size, output_dim, dropout_p=0.1, max_length=X.size()[1])

    if use_cuda:
        rnn = decoder1.cuda()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder1.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = zip(X, Y)
    if loss_function == "cosine":
        criterion = nn.CosineEmbeddingLoss()
    else:
        print("ERROR: need to add loss type")

    for ep in range(num_epochs):
        print('epoch ', str(ep))
        for iter in range(1, n_iters + 1):
            print("iter: ", str(iter))
            training_pair = training_pairs[iter-1]
            input_variable = Variable(training_pair[0])
            target_variable = Variable(training_pair[1])

    print("NOT IMPLEMENTED")


''' Create and train a CNN model
    Hybrid features supported - pass structured feats as X2
    Does NOT support joint training yet
    returns: the CNN model
'''
def cnn_model(X, Y, act=None, windows=[1,2,3,4,5], X2=[], num_epochs=10, loss_func='categorical_crossentropy'):
    #Y = to_categorical(Y)
    #X = numpy.asarray(X)
    #embedding_size = X.shape[-1]
    #max_seq_len = X.shape[1]
    #print "train X shape: " + str(X.shape)
    #print "max_seq_len: " + str(max_seq_len)
    #inputs = []
    #input_arrays = [X]
    #hybrid = False

    # Train the CNN, return the model
    st = time.time()
    Xarray = numpy.asarray(X).astype('float')
    Yarray = Y.astype('int') #Y.toarray().astype('int')
    print("X numpy shape: ", str(Xarray.shape), "Y numpy shape:", str(Yarray.shape))

    # Params
    X_len = Xarray.shape[0]
    dim = Xarray.shape[-1]
    num_labels = Yarray.shape[-1]
    num_epochs = 10
    steps = 0
    best_acc = 0
    last_step = 0
    log_interval = 1000
    batch_size = 100
    num_batches = math.ceil(X_len/batch_size)
    learning_rate = 0.001

    cnn = CNN_Text(dim, num_labels)
    if use_cuda:
	cnn = cnn.cuda()

    # Train
    optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

    steps = 0
    cnn.train()
    for epoch in range(num_epochs):
	print("epoch", str(epoch))
        i = 0
        numpy.random.seed(seed=1)
        permutation = torch.from_numpy(numpy.random.permutation(X_len)).long()
        Xiter = Xarray[permutation]
        Yiter = Yarray[permutation]

        while i+batch_size < X_len:
             batchX = Xiter[i:i+batch_size]
             batchY = Yiter[i:i+batch_size]
             Xtensor = torch.from_numpy(batchX).float()
             Ytensor = torch.from_numpy(batchY).long()
             if use_cuda:
                  Xtensor = Xtensor.cuda()
                  Ytensor = Ytensor.cuda()
             feature = Variable(Xtensor)
             target = Variable(Ytensor)
             i = i+batch_size

             optimizer.zero_grad() 
             logit = cnn(feature)

             loss = F.cross_entropy(logit, torch.max(target,1)[1])
             loss.backward()
             optimizer.step()

             steps += 1
 
        # Print epoch time
        ct = time.time() - st
        unit = "s"
        if ct > 60:
	    ct = ct/60
            unit = "m"
        print("time so far: ", str(ct), unit)
    return cnn

def test_cnn(model, testX, testids, probfile='/u/yoona/data/torch/probs_win200_epo10', labelencoder=None, collapse=False):
    y_pred = []
    y_pred_softmax = []
    y_pred_logsoftmax = []
    softmax = nn.Softmax(dim=1)
    logsoftmax = nn.LogSoftmax(dim=1)

    new_y_pred = []
    if probfile is not None:
        probs = []

    for x in range(len(testX)):
        input_row = testX[x]
   
        icd = None
        if icd is None:
            input_tensor = torch.from_numpy(numpy.asarray([input_row]).astype('float')).float()
            if use_cuda:
                input_tensor = input_tensor.contiguous().cuda()
            icd_var = model(Variable(input_tensor))
            # Softmax and log softmax values
            icd_vec = logsoftmax(icd_var)
            icd_vec_softmax = softmax(icd_var)
            icd_code = torch.max(icd_vec, 1)[1].data[0]
           
	    icd_prob = torch.max(icd_vec_softmax,1)[0]
	    if icd_prob < threshold:
		new_y_pred.append(9)
	    else:
		new_y_pred.append(icd_code)
	
            # Save the probabilties
            if probfile is not None:
                probs.append(icd_prob)

        y_pred.append(icd_code)

    if probfile is not None:
        pickle.dump(probs, open(probfile, "wb"))

    print "Probabilities: " + str(probs)

    #return y_pred
    return new_y_pred


def train_encoder_decoder(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, loss_func, max_length):
    # Parameters
    teacher_forcing_ratio = 0.5
    loss = 0

    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]
    target_length = target_variable.size()[0]
    #print "input_length: " + str(input_length) + ", target_length: " + str(target_length)
    input_dim = input_variable.size()[-1]
    output_dim = target_variable.size()[-1]
    start_token = torch.zeros(output_dim)
    end_token = Variable(torch.zeros(output_dim))
    loss_target = Variable(torch.ones(1))

    #print "start token size: " + str(start_token.size())

    encoder_outputs = Variable(torch.zeros(input_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
    for ei in range(input_length):
        #print "ei: " + str(ei)
        #print "encoder_hidden: " + str(encoder_hidden.data.size())
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(start_token)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input
    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(target_length):
            #print "di (tf): " + str(di)
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            target_var = target_variable[di].view([1,-1])
            #print "- decoder_ouput: " + str(decoder_output.size())
            #print "- target_variable[di]: " + str(target_var.size())
            loss += loss_func(decoder_output, target_var, loss_target)
            decoder_input = target_variable[di]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(target_length):
            #print "di: "
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            #print "decoder_output: " + str(decoder_output)
            #topv, topi = decoder_output.data.topk(1)
            #ni = topi[0][0]
            #print "ni: " + str(type(ni)) + " : " + str(ni)

            #decoder_input = Variable(torch.FloatTensor([[ni]]))
            decoder_input = decoder_output
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            target_var = target_variable[di].view([1,-1])
            #print "- decoder_ouput: " + str(decoder_output.size())
            #print "- target_variable[di]: " + str(target_var.size())

            loss += loss_func(decoder_output, target_var, loss_target)
            if torch.equal(decoder_output, end_token):
                break

    print("loss: ", str(loss))

    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length

def encoder_decoder_model(X, Y, num_nodes, print_every=1000, plot_every=100, learning_rate=0.01, num_epochs=10, loss_function="cosine"):
    print("Encoder-decoder model: ", str(num_nodes))
    n_iters = len(X)
    print("epochs: ", str(num_epochs), " iters: ", str(n_iters))
    if use_cuda:
        print("using cuda")
        X = torch.cuda.FloatTensor(X)
        Y = torch.cuda.FloatTensor(Y)
    else:
        X = torch.FloatTensor(X)
        Y = torch.FloatTensor(Y)
    print("X: ", str(X.size()))
    print("Y: ", str(Y.size()))
    hidden_size = num_nodes
    input_dim = X.size()[-1]
    max_length = Y.size()[1]
    output_dim = Y.size()[-1]
    print("input_dim: " + str(input_dim))
    print("max_length: " + str(max_length))
    print("output_dim: " + str(output_dim))
    encoder1 = EncoderRNN(input_dim, hidden_size)
    decoder1 = AttnDecoderRNN(hidden_size, output_dim, dropout_p=0.1, max_length=X.size()[1])


    if use_cuda:
        encoder1 = encoder1.cuda()
        decoder1 = decoder1.cuda()

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder1.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder1.parameters(), lr=learning_rate)
    #training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]
    training_pairs = zip(X, Y)
    if loss_function == "cosine":
        criterion = nn.CosineEmbeddingLoss()
    else:
        print("ERROR: need to add loss type")

    for ep in range(num_epochs):
        print('epoch ', str(ep))
        for iter in range(1, n_iters + 1):
            print("iter: ", str(iter))
            training_pair = training_pairs[iter-1]
            input_variable = Variable(training_pair[0])
            target_variable = Variable(training_pair[1])

            #print "input var: " + str(input_variable.data.size())
            #print "target var: " + str(target_variable.data.size())

            loss = train_encoder_decoder(input_variable, target_variable, encoder1, decoder1, encoder_optimizer, decoder_optimizer, criterion, max_length)
            print_loss_total += loss
            plot_loss_total += loss

            #if iter % print_every == 0:
            #    print_loss_avg = print_loss_total / print_every
            #    print_loss_total = 0
            #    print('%s (%d %d%%) %.4f' % (timeSince(start, float(iter / n_iters)), iter, float(iter / n_iters) * 100, print_loss_avg))

            #if iter % plot_every == 0:
            #    plot_loss_avg = plot_loss_total / plot_every
            #    plot_losses.append(plot_loss_avg)
            #    plot_loss_total = 0

    return encoder1, decoder1, output_dim

'''
    testX: python list of input sequences (python lists)
    testY: python list of output sequences (also python lists)
    max_length: the maximum sequence length
'''
def test_encoder_decoder(encoder, decoder, testX, max_length, output_dim):
    predY = []
    for test_seq in testX:
        decoded_seq, decoded_attn = decode(encoder, decoder, test_seq, max_length, output_dim)
        pred_seq = decoded_seq.storage.tolist()
        predY.append(pred_seq)
    return predY

def decode(encoder, decoder, input_vec, max_length, output_dim):
    if use_cuda:
        input_variable = Variable(torch.cuda.FloatTensor(input_vec))
    else:
        input_variable = Variable(torch.FloatTensor(input_vec))
    input_length = input_variable.size()[0]
    input_dim = input_variable.size()[-1]
    start_token = torch.zeros(input_dim)
    end_token = torch.ones(output_dim)
    encoder_hidden = encoder.initHidden()

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

    decoder_input = Variable(start_token)  # SOS
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden
    decoded_seq = []
    decoder_attentions = torch.zeros(max_length, max_length)

    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)

        decoder_attentions[di] = decoder_attention.data
        topv, topi = decoder_output.data.topk(1)
        ni = topi[0][0]
        if ni == end_token:
            decoded_seq.append(ni)
            break
        else:
            decoded_seq.append(ni)

        decoder_input = Variable(torch.FloatTensor([[ni]]))
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    return decoded_seq, decoder_attentions[:di + 1]

##################################################
# HELPER FUNCTIONS
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

def showAttention(input_sentence, output_words, attentions):
    print("TODO")
    # Set up figure with colorbar
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #cax = ax.matshow(attentions.numpy(), cmap='bone')
    #fig.colorbar(cax)

    # Set up axes
    #ax.set_xticklabels([''] + input_sentence.split(' ') + ['<EOS>'], rotation=90)
    #ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    #ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    #ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    #plt.show()

def evaluateAndShowAttention(input_sentence, encoder, attn_decoder):
    output_words, attentions = decode(encoder, attn_decoder, input_sentence, 10, 100)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)
