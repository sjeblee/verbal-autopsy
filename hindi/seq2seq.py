# coding: utf-8
from __future__ import unicode_literals, print_function, division


# In[45]:

#get_ipython().magic('matplotlib inline')


# In[46]:
import sys
sys.path.append('/u/sjeblee/research/git/verbal-autopsy')
sys.path.append('/u/sjeblee/research/git/gensim')
from gensim.models import KeyedVectors
import word2vec

from io import open
import unicodedata
import string
import re
import random
import time

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

#use_cuda = torch.cuda.is_available()
#if use_cuda
device = torch.device("cuda") # if torch.cuda.is_available() else "cpu")
print("device:", device)
debug = False

start_time = time.time()

# In[47]:

SOS_token = 0
EOS_token = 1
tgt_emb_file = '/u/sjeblee/research/vectors/wiki.hi.vec'
src_emb_file = '/u/sjeblee/research/vectors/GoogleNews-vectors-negative300.bin'


#print('loading target word vectors...')
#tgt_vec_model = KeyedVectors.load_word2vec_format(tgt_emb_file, binary=False, unicode_errors='ignore')
print('loading source word vectors...')
src_vec_model = KeyedVectors.load_word2vec_format(src_emb_file, binary=True, unicode_errors='ignore')


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split():
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[48]:

# Turn a Unicode string to plain ASCII, thanks to
# http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters


def normalizeString(s):
    
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s.lower()


# In[55]:

def readLangs(lang1, lang2,en_file,hi_file):
    print("Reading lines...")

    # Read the file and split into lines
    lines_en = open(en_file, encoding='utf-8').read().strip().lower().split('\n')
    lines_hi = open(hi_file, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang,lines_en,lines_hi


# In[56]:

MIN_LENGTH = 3
MAX_LENGTH = 50

def filterPair(l):
    return len(l.split()) < MAX_LENGTH


def filterPairs(lines, lines2):
    #for y in range(len(lines)):
    #    words = lines[y].split()
    #    if len(words) > MAX_LENGTH:
    #        words = words[0:MAX_LENGTH]
    #    lines[y] = ' '.join(words)
    # Remove short lines
    if debug: print('filtering pairs...')
    y = 0
    while y < len(lines):
        words = lines[y].split()
        if len(words) < MIN_LENGTH or len(words) > (MAX_LENGTH-1):
            del lines[y]
            del lines2[y]
            y = y-1
        y += 1
    y = 0
    while y < len(lines2):
        words = lines2[y].split()
        if len(words) < MIN_LENGTH or len(words) > (MAX_LENGTH-1):
            del lines[y]
            del lines2[y]
            y = y-1
        y += 1
    return lines, lines2


# In[63]:

def prepareData(lang1, lang2,en_file,hi_file):
    input_lang, output_lang, lines_en,lines_hi = readLangs(lang1, lang2,en_file,hi_file)
    print("Read %s enlish sentences" % len(lines_en))
    print("Read %s hindi sentences" % len(lines_hi))
    lines_en, lines_hi = filterPairs(lines_en, lines_hi)
    print("Trimmed to %s sentence pairs" % len(lines_en))
    print("Counting words...")
    for l in lines_en:
        input_lang.addSentence(l)
    for l in lines_hi:
        output_lang.addSentence(l)
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    zipped = zip(lines_en,lines_hi)
    pairs = list(zipped)
    return input_lang, output_lang, pairs


input_lang, output_lang, pairs = prepareData('eng', 'hi','/u/sjeblee/research/data/hindi/parallel/IITB.en-hi.en','/u/sjeblee/research/data/hindi/parallel/IITB.en-hi.hi')
pair = random.choice(pairs)
print(pair[0])
print (pair[1].encode(encoding='UTF-8',errors='ignore'))


# In[82]:

pair = random.choice(pairs)
print(pair[0])
print(pair[1].encode(encoding='UTF-8',errors='ignore'))


# In[75]:

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        #self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        #embedded = self.embedding(input).view(1, 1, -1)
        embedded = input.view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[26]:

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[27]:

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[28]:

def indexesFromSentence(lang, sentence):
    indices = [lang.word2index[word] for word in sentence.split()]
    #if debug: print("target words:", str(len(indices)))
    indices.append(EOS_token)
    while len(indices) < MAX_LENGTH:
        indices.append(EOS_token)
    return indices

def vectorsFromSentence(lang, sentence, vec_model):
    vecs = [word2vec.get(word, vec_model) for word in sentence.split()]
    vecs.append(torch.zeros(vec_model.vector_size)) # EOS token
    while len(vecs) < MAX_LENGTH:
        vecs.append(torch.zeros(vec_model.vector_size)) # pad the sequence
    return vecs
    #return [lang.word2index[word] for word in sentence.split(' ')]


def tensorFromSentence(lang, sentence, vec_model):
    indexes = indexesFromSentence(lang, sentence, vec_model)
    if len(indexes) > MAX_LENGTH-1:
        indexes = indexes[0:MAX_LENGTH-1]
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.float, device=device).view(-1, 1)


def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0], src_vec_model)
    target_tensor = tensorFromSentence(output_lang, pair[1], tgt_vec_model)
    return (input_tensor, target_tensor)

def tensorsFromPairs(pairs):
    src_seqs = []
    tgt_seqs = []
    batch_size = len(pairs)
    for pair in pairs:
        src_vec = vectorsFromSentence(input_lang, pair[0], src_vec_model)
        #tgt_vec = indexesFromSentence(output_lang, pair[1], tgt_vec_model)
        tgt_vec = indexesFromSentence(output_lang, pair[1])
        src_seqs.append(src_vec)
        tgt_seqs.append(tgt_vec)
        #if debug: print('src_vec:', str(len(src_vec)), 'tgt_vec:', str(len(tgt_vec)))
    src_tensor = torch.tensor(src_seqs, dtype=torch.float, device=device)#.view(-1, 1)
    tgt_tensor = torch.tensor(tgt_seqs, dtype=torch.long, device=device).view(batch_size, -1, 1)
    return src_tensor, tgt_tensor

# In[29]:

teacher_forcing_ratio = 0.5


def train(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #print("X: ", str(input_batch.size()), " Y: ", str(target_batch.size()))
    batch_size = input_batch.size(0)
    
    for x in range(batch_size):
        input_tensor = input_batch[x]
        target_tensor = target_batch[x]
        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                loss += criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == EOS_token:
                    break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_vec = vectorsFromSentence(input_lang, sentence, src_vec_model)
        input_tensor = torch.tensor(input_vec, dtype=torch.float, device=device)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


# In[34]:

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')

# In[30]:

import time
import math


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


# In[31]:

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    batch_size = 1
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs) for i in range(n_iters*batch_size)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        start = (iter-1)*batch_size
        end = iter*batch_size
        if debug: print("start: ", str(start), "end:", str(end))
        input_tensor, target_tensor = tensorsFromPairs(training_pairs[start:end])
        #input_tensor = training_pair[0]
        #target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        # Free the tensor memory
        del input_tensor
        del target_tensor

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

            print('saving models...')
            prefix = "/u/sjeblee/research/data/hindi/en_hi_"
            torch.save(encoder, prefix + "encoder.model")
            torch.save(decoder, prefix + "decoder.model")
            evaluateRandomly(encoder, decoder, n=1)
        #if iter % plot_every == 0:
        #    plot_loss_avg = plot_loss_total / plot_every
        #    plot_losses.append(plot_loss_avg)
        #    plot_loss_total = 0

    #showPlot(plot_losses)


# In[32]:

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


# In[33]:

# In[38]:

def evaluateRandomly2(encoder, decoder, in_string):
        print('>', in_string)
        print('=')
        output_words, attentions = evaluate(encoder, decoder, in_string)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[35]:

hidden_size = 300
encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 1000000, print_every=1000)


# In[36]:

evaluateRandomly(encoder1, attn_decoder1)


# In[44]:

evaluateRandomly2(encoder1, attn_decoder1,"I am good")

print('total time:', asMinutes(time.time()-start_time), 'mins')
# In[ ]:



