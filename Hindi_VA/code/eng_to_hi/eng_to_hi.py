
# coding: utf-8

# In[2]:

get_ipython().magic('matplotlib inline')


# In[1]:

from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:",device)


# In[2]:

pad_token = 0 
SOS_token = 1
EOS_token = 2


class Lang:
    def __init__(self, name,max_seq_length):
        self.name = name
        self.max_seq_len = max_seq_length
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS",2:"EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


# In[3]:

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
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s


# In[4]:

MAX_LENGTH = 50
def readLangs(lang1, lang2,en_file,hi_file):
    print("Reading lines...")
    
    max_length_hi=0
    max_length_eng = 0
    longest_eng=''
    longest_hi = ''
    # Read the file and split into lines
    lines_en = open(en_file, encoding='utf-8').        read().strip().split('\n')
    lines_hi = open(hi_file, encoding='utf-8').        read().strip().split('\n')
    y=0
    
    while y < len(lines_en):
        words = lines_en[y].split(" ")
        if len(words) > (MAX_LENGTH):
            del lines_en[y]
            del lines_hi[y]
            y=y-1
        y+=1
        
    y=0
    while y < len(lines_hi):
        words = lines_hi[y].split(" ")
        if len(words) > (MAX_LENGTH):
            del lines_hi[y]
            del lines_en[y] 
            y=y-1
        y+=1
    for l in lines_en:
        words = l.split(" ")
        if len(words)>max_length_eng:
            max_length_eng = len(words)
            longest_eng = l
    for li in lines_hi:
        words = li.split(" ")
        if len(words)>max_length_hi:
            max_length_hi = len(words)
            longest_hi = li
    
    print("longest english: %s"%longest_eng)
    print("longest hindi:%s"%longest_hi)

    # Split every line into pairs and normalize
    #pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
   

    input_lang = Lang(lang1,max_length_eng)
    output_lang = Lang(lang2,max_length_hi)

    return input_lang, output_lang,lines_en,lines_hi


# In[5]:



def filterPair(l):
    return len(l.split(' ')) < MAX_LENGTH


def filterPairs(lines):
    return [l for l in lines if filterPair(l)]


# In[6]:


def prepareData(lang1, lang2,en_file,hi_file):
    
    input_lang, output_lang, lines_en,lines_hi = readLangs(lang1, lang2,en_file,hi_file)
    print("Read %s enlish sentences" % len(lines_en))
    print("Read %s hindi sentences" % len(lines_hi))
#     lines_en = filterPairs(lines_en)
#     lines_hi = filterPairs(lines_hi)
#     print("Trimmed to %s sentence pairs" % len(lines_en))
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


input_lang, output_lang, pairs = prepareData('eng', 'hi','./parallel/IITB.en-hi.en','./parallel/IITB.en-hi.hi')
pair = random.choice(pairs)
print(pair[0])
print (pair[1].encode(encoding='UTF-8',errors='ignore'))
print("max_english: %d"%input_lang.max_seq_len)
print("max_hindi: %d"%output_lang.max_seq_len)


# In[7]:

pair = random.choice(pairs)
print(pair[0])
print (pair[1].encode(encoding='UTF-8',errors='ignore'))


# In[19]:

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[9]:

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


# In[10]:

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
        print("ATTN encoder _outputs size: {}".format(encoder_outputs.size()))
        embedded = self.embedding(input).view(1, 1, -1)
        print("ATTN embedded: {}".format(embedded.size()))
        
        embedded = self.dropout(embedded)
        print("ATTN embedded after dropout size: {}".format(embedded.size()))

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        print("ATTN input before attn_weights: {}".format(torch.cat((embedded[0], hidden[0]), 1).size()))
        print("ATTN attn_weights: {}".format(attn_weights.size()))
        
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        print("ATTN attn applied size: {}".format(attn_applied.size()))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        print("ATTN output: {}".format(output.size()))
        
        output = self.attn_combine(output).unsqueeze(0)
        print("ATTN attn combine output size: {}".format(output.size()))

        output = F.relu(output)
        print("ATTN ouput after relu: {}".format(output.size()))
        output, hidden = self.gru(output, hidden)
        print("ATTN output after gru size: {}".format(output.size()))
        print("ATTN hidden after gru size: {}".format(hidden.size()))

        output = F.log_softmax(self.out(output[0]), dim=1)
        print("ATTN output after softmax: {}".format(output.size()))
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)


# In[11]:



def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    for i in range (lang.max_seq_len-len(indexes)):
        indexes.append(pad_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def tensorsFromPairs(pairs):
    
    batch_size = len(pairs)
    src_seqs = torch.tensor((),dtype=torch.long,device=device)
    tgt_seqs = torch.tensor((),dtype=torch.long,device=device)
    for pair in pairs:
        input_tensor = tensorFromSentence(input_lang, pair[0])
        target_tensor = tensorFromSentence(output_lang, pair[1])
        src_seqs = torch.cat([src_seqs,input_tensor])
        tgt_seqs = torch.cat([tgt_seqs,target_tensor])
    src_seqs = src_seqs.view(batch_size,-1,1)
    tgt_seqs = tgt_seqs.view(batch_size,-1,1)
    return src_seqs, tgt_seqs
    

pair_1 = random.choice(pairs)
pair_2 = random.choice(pairs)
p = [pair_1,pair_2]
print(pair_1[0])
print (pair_1[1].encode(encoding='UTF-8',errors='ignore'))
print(pair_2[0])
print (pair_2[1].encode(encoding='UTF-8',errors='ignore'))
input_t,output_t = tensorsFromPairs(p)
print(input_t.size())
print(output_t.size())


# In[12]:

teacher_forcing_ratio = 0.5


def train(input_batch, target_batch, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = input_batch.size(0)
#     input_length = input_tensor.size(0)
#     target_length = target_tensor.size(0)

    

    

    for x in range(batch_size):
        
        input_tensor = input_batch[x]
        print("input_tensor size: {}".format(input_tensor.size()))
        print("input_tensor[0] size: {}".format(input_tensor[0].size()))
        print("input_tensor[0]: {}".format(input_tensor[0]))
        
        target_tensor = target_batch[x]
        print("target_tensor size: {}".format(target_tensor.size()))
        
        input_length = input_tensor.size(0)
        print("input_length size: {}".format(input_length))
        
        target_length = target_tensor.size(0)
        print("target_length size: {}".format(target_length))
        
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        print("encoder output size: {}".format(encoder_outputs.size()))
        
        loss = 0
        for ei in range(input_length):
            
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
#             print("encoder output size: {}".format(encoder_output.size()))
#             print("encoder hidden size: {}".format(encoder_hidden.size()))
#             print("encoder output size: {}".format(encoder_output[0,0].size()))
            
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


# In[13]:

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


# In[14]:

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.001):
    start = time.time()
    batch_size = 128
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [random.choice(pairs)
                      for i in range(n_iters*batch_size)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        start = (iter-1)*batch_size
        end = iter*batch_size
        input_tensor, target_tensor = tensorsFromPairs(training_pairs[start:end])
#         input_tensor = training_pair[0]
#         target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
            prefix = "./"
            torch.save(encoder, prefix + "encoder.model")
            torch.save(decoder, prefix + "decoder.model")
            evaluateRandomly(encoder, decoder, n=1)

#         if iter % plot_every == 0:
#             plot_loss_avg = plot_loss_total / plot_every
#             plot_losses.append(plot_loss_avg)
#             plot_loss_total = 0

#     showPlot(plot_losses)


# In[15]:

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


# In[16]:

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
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


# In[17]:

def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[38]:

def evaluateRandomly2(encoder, decoder, in_string):
        print('>', in_string)
        print('=')
        output_words, attentions = evaluate(encoder, decoder, in_string)
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')


# In[20]:

if __name__ == '__main__':
    hidden_size = 256
    encoder1 = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    attn_decoder1 = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=0.1).to(device)

    trainIters(encoder1, attn_decoder1, 500, print_every=50)


# In[ ]:



