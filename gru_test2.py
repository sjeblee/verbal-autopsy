#!/usr/bin/python
# Build a classifier model with the VA features
# @author sjeblee@cs.toronto.edu

import sys
sys.path.append('../keras-attention-mechanism')
sys.path.append('keywords')
import numpy
import time
from keras.utils.np_utils import to_categorical
from sklearn import metrics
from sklearn import preprocessing
from sklearn.metrics import f1_score
#import attention_utils
import cluster_keywords
import data_util
import rebalance
import math
#import model_lib_test
#from layers import Attention
from lxml import etree
import os
import pickle
import re


import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
labelencoder_adult = None
labelencoder_child = None
labelencoder_neonate = None
vec_types = ["narr_vec", "narr_seq", "event_vec", "event_seq", "symp_vec", "kw_vec", "textrank_vec"]
numpy.set_printoptions(threshold=numpy.inf)

# Output top K features
output_topk_features = True

'''
be aware that dropout is not used here (?why)
The difference compared with the 

'''
def main():
    global cuda
    cuda = torch.device("cuda:0")
    train_mode = True     
    total_start_time = time.time()
    # Params
    arg_rebalance = ""
    arg_anova = "chi2"
    arg_train_feats = "u/yanzhaod/data/va/mds+rct/train_child_cat_spell.xml"
    arg_test_feats = "u/yanzhaod/data/mds+rct/test_child_cat_spell.xml"
    global max_num_char
    global all_categories
    global n_hidden
    global emb_dim_char
    global n_iters
    global vocab
    vocab = 'abcdefghijklmnopqrstuvwxyz0123456789 '
    train_mode = True
    
    n_hidden = 64
    emb_dim_char = 30
    batch_size = 128
    model_out="gru_child.pt"
    out_results = 'out_child_results.txt'
    total_start_time = time.time()

    if train_mode == True:
        training_generator = preprocess(arg_train_feats,"train",batch_size=batch_size)
        torch.save(training_generator,"training_generator.pt")
        gru = train_gru(training_generator,max_num_char,n_hidden,all_categories,emb_dim_char,model_out=model_out)
    else:
        training_generator = torch.load("training_generator.pt")
        gru = torch.load(model_out)
        
    test_gru(gru,arg_test_feats,out_results)


def timeSince(since):
   now = time.time()
   s = now - since
   m = math.floor(s / 60)
   s -= m * 60
   return '%dm %ds' % (m, s)
def save(m,out_model_filename):
    torch.save(m, out_model_filename)
    print('Model saved as %s' % out_model_filename)
    
def writeToFile(line,filename):
   if os.path.exists(filename):
           append_write = 'a' # append if already exists
   else:
           append_write = 'w' # make a new file if not

   f = open(filename,append_write)
   f.write(line + '\n')
   f.close()
def categoryFromOutput(output):
    top_n, top_i = output.topk(1)
    #print(top_n)
    #print(top_i.size())
    category_i = top_i[0].item()
    #print(category_i)
    return all_categories[category_i], category_i
def lineToTensorChar(narrative):
    tensor = torch.zeros([max_num_char,1],device=cuda)
    for li, letter in enumerate(narrative):
        try:
            tensor[li][0] = vocab.index(letter)
        except ValueError:
            print("letter %s not found"%letter)
    return tensor
def getTensorsChar(category,line):
    category_tensor = torch.tensor(all_categories.index(category), dtype=torch.long,device=cuda)
    line_tensor = lineToTensorChar(line)
    category_tensor = category_tensor.to(cuda)
    return category, line, category_tensor, line_tensor
def preprocess(input_train,mode,batch_size=128):
    if mode != "train" and mode != "test":
        print("Error: mode must be either 'train' or 'test'")
        return None
    print("Preprocessing...")
    stime = time.time()
    data  ={}
    tree = etree.parse(input_train)

    global all_categories
    global vocab
    if mode == "train":
        global max_num_char
        global n_hidden
        global n_iters
        all_categories = []
        for e in tree.iter("cghr_cat"):
            text = e.text.lower()
            if text not in data:
                 data[text]=[]
                 all_categories.append(text)
        print("vocab: %s" %vocab)
    elif mode == "test":
        for cat in all_categories:
            data[cat] = []
    root = tree.getroot()
    for child in root:
        MG_ID = child.find('MG_ID')
        narrative = child.find('narrative')
        cghr_cat = child.find('cghr_cat')
        cghr_cat2 = [child.find('CODINGKEYWORDS1'),child.find('CODINGKEYWORDS2')]
        second_try = []
        try:
            text = narrative.text.lower()
    
        except AttributeError:
            
            for e in cghr_cat2:
                try:
                    second_try.append(child.find('CODINGKEYWORDS1').text.lower())
                except AttributeError:
                    continue
            if len(second_try) == 2:
                if second_try[0] == second_try[1]:
                    second_try = second_try[0]
                else:
                    second_try = ' '.join(second_try)
            elif len(second_try) == 1:
                second_try = second_try[0]
            else:
                print("undetected mgid: "+MG_ID.text)
        if second_try:
            text = text + ' ' + second_try.lower()
            #print(MG_ID.text)
        text = re.sub('[^a-z0-9\s]','',text)           #Note:this steps removes all punctionations and symbols in text, which is optional
        data[cghr_cat.text].append((MG_ID.text,text))


    n_iters = 0
    for k,v in data.items():
        n_iters += len(v)

    print("number of the narratives: %d" %n_iters)

    l = []
    if mode == "train":
        max_num_char = 0
        for k in data:
            v = data[k]
            for i in range(len(v)):
                if len(v[i][1]) > max_num_char:
                    max_num_char = len(v[i][1])
                l.append((k,(v[i][0],v[i][1])))
    elif mode == "test":
        for k,v in data.items():
            for i in range(len(v)):
                l.append((k,(v[i][0],v[i][1])))
    training_set_char = []
    for e in l:
        k,v = e[0],e[1]
        category, line, category_tensor, line_tensor_char = getTensorsChar(k,v[1])
        line_tensor_char = line_tensor_char.to(cuda)
        #print(line_tensor_char.size())
        training_set_char.append([category_tensor,line_tensor_char,category,v[0]])
        training_params = {"batch_size": batch_size,
                        "shuffle": True,
                        "num_workers": 0}
    training_generator = DataLoader(training_set_char, **training_params)

    print("Preprocessing took %s"%timeSince(stime))

    return training_generator

def train_gru(training_generator,max_num_char,n_hidden,all_categories,emb_dim_char,epochs=30,learning_rate=0.001,model_out="gru_child.pt"):
    gru = GRU_Text(max_num_char,n_hidden,len(all_categories),emb_dim_char)
    gru.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer_char = torch.optim.Adam(gru.parameters(), lr=learning_rate)
    batch_size = 128
    print_every = 21
#    current_loss = 0
#    all_losses = []
    
    print("training...")
    stime = time.time()

    count = 0
    for epoch in range(epochs):
        for iter, batch in enumerate(training_generator):
            count += 1
            category_tensor, line_tensor_char,category,MG_ID= batch
            if line_tensor_char.size() == (0,):
                continue 
            #train
            optimizer_char.zero_grad() 
            print(line_tensor_char.size())
            output,hidden = gru(line_tensor_char,None)
            loss = criterion(output, category_tensor)
            loss.backward()
            optimizer_char.step()
            if count % print_every == 0:
                print('%d %d%% (%s)' % (count*batch_size, count*batch_size / n_iters/epochs*100, timeSince(stime)))  
    save(gru,model_out)
    print("Training took %s"%timeSince(stime))
    return gru



class GRU_Text(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_size):
        super(GRU_Text, self).__init__()
        
        self.hidden_size = hidden_size
        
        self.encoder = nn.Embedding(input_size,emb_size)
        self.gru = nn.GRU(emb_size*input_size,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        '''
        input is of size (batch,max number of characters in a narrative,char_)
        '''
        input = self.encoder(input.long())    #(batch,num,1,emb)
        input = input.squeeze(2)   #(batch,num,emb)
        input = input.view(input.size(0),input.size(1)*input.size(2)) #(batch,num*emb)
        input = input.unsqueeze(0)    #(1,batch,num*emb)
        b,hidden = self.gru(input,hidden)  #(1,batch,hidden)
#        print(b.size())
        b = self.linear(b) #(1,batch,output_size)
        #output = self.softmax(output)
        b = b.squeeze(0)
        return b, hidden
        
    def initHidden(self):
        return torch.zeros([1, self.hidden_size],device=cuda)
    
def test_gru(model,input_test,out_results):
    testing_generator = preprocess(input_test,"test")
    result = []
    cat_pred,cat_true = [],[]
    start = time.time()
    for iter, batch in enumerate(testing_generator):
        category_tensor, line_tensor_char,category,MG_ID= batch
        if line_tensor_char.size() == (0,):
            continue 
        output,hidden = model(line_tensor_char,None)
        
        print("category",category)
        l = []
        for i in range(output.size(0)):
            guess,guess_i = categoryFromOutput(output[i])
            l.append(guess)
#        guess, guess_i = categoryFromOutput(output)
        guess=l
        print("output",l)
        result.append({'Correct_ICD':category,'Predicted_ICD':guess,'MG_ID':MG_ID})
        cat_pred += list(guess)
        cat_true += category
    print('----------------------------------------------')
    f1score = f1_score(cat_true,cat_pred,average="weighted")
    print(f1score)
#    writeToFile("f1score: " + str(f1score),out_text)
    for i in range(len(result)):
        result[i] = str(result[i])
    writeToFile('\n'.join(result),out_results)
    return

def split_feats(keys, labelname):
    ignore_feats = ["WB10_codex", "WB10_codex2", "WB10_codex4"] 
    vec_keys = [] # vector/matrix features for CNN and RNN models
    point_keys = [] # traditional features for other models
    for key in keys:
        if key in vec_types:
            vec_keys.append(key)
        elif key == labelname or key not in ignore_feats:
            point_keys.append(key)

    print("vec_keys: " + str(vec_keys))
    print("point_keys: " + str(point_keys))
    print("Keys printed")
    return vec_keys, point_keys

#########################################################
# Select K Best symptoms for each class
# Create output files containing best k features for each class
# Arguments
# 	X		: list of features
# 	Y		: ndarray after labelencoder transformation
# 	function	: type of anova function (ex. f_classif, chi2)
#	output_path	: path to the output file
# 	k		: number of top-k features to be selected
#
#
def select_top_k_features_per_class(X, Y, function, output_path, k = 100):
    classes = labelencoder.classes_

    for i in range(len(classes)):
        output = open(output_path + "/top_" + str(k) + "_features_class_" + classes[i], 'w')
        output.write("Class : " + str(classes[i]))
        print("Class: " + str(classes[i]))
        this_Y = []
        for j in range(len(Y)):
            if Y[j] == i:
                binary= 1
            else:
                binary= 0
            this_Y.append(binary)
        anova_symp = SelectKBest(function, 'all')
        anova_symp.fit(X,this_Y)
        best_indices = anova_symp.get_support(True)
        scores = anova_symp.scores_
        output.write("The sorted indices:")
        sorted_idx = numpy.argsort(scores)[::-1][:k]
        output.write(str(sorted_idx))
    
        for i in range(len(sorted_idx)):
            selected = str(keys[sorted_idx[i] + 2])
            put.write("\n")
            put.write(selected + " ")
            output.write(str(scores[sorted_idx[i]]))
        output.close()

if __name__ == "__main__":main()
