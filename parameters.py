import os
if os.path.exists('D:/projects/Git/verbal-autopsy'):
    USE_SERVER = False
else:
    USE_SERVER = True   # To run in CSLab server, otherwise home computer
#-----------------------mode----------------------------
TRAINING = True       # To train the model; otherwise load existing model

DEBUG = True            # Debug mode
PRE_DATA = True     # To preprocess the input data, otherwise load existing numpy array 
CROSS_VAL = True        # To do cross validation on the ten-fold split sets (need to train and load data each time)
embedding = 'struct-elmo'       # Choices: 'conc', 'elmo', 'char','struct','struct-elmo','bert','struct-bert'
model_type = 'cnn'       #Choices: 'cnn','lstm','gru'
rec_type = "child"      # Choices: 'child','neonate','adult'
dataset = 'mds'       # Choices: 'mds','phmrc'
#preprocess = 'stem lemmatize ' #Choices of one or more following words in the list 'stem' ,'lemmatize','backward_padding','no_punc'
preprocess = 'backward_padding ' 
module = ''   #attention
#---------------------parameters-----------------------
rep = 3
learning_rate = 0.001    #0.003
num_epochs = 10         #12 #13
batch_size= 120
emb_dim_word = 100
max_num_word = 200     #maximum allowed number of wards in a narrative text  #200/220
emb_dim_char= 24       #embedding dimension of character, other dimensions are tenprarily unavailable
max_char_in_word = 7   #maximum number of characters counted in a word, which is used in input_conc model
max_num_char = 1000    #maximum number of characters allowed in a text, which is used in character-embedding model
dropout = 0.3
feat = ['region','language']    #can be either a feature string or a list of feature strings
                                #current feature: 'region','Language','StateCode','DeathPlace','Religion','Education','Marital','DeathAge'
#---------------------bert parameters-----------------------
BERT_MODEL = 'pretrained_bert_tf/biobert_pretrain_output_all_notes_150000' 
#BERT_MODEL = 'pretrained_bert_tf/biobert_pretrain_output_disch_100000'
#BERT_MODEL = 'biobert_v1.1_pubmed'


VOCAB_FILE = 'weights/'+BERT_MODEL+'/vocab.txt'
BERT_CONFIG_FILE = 'weights/'+BERT_MODEL+'/bert_config.json'
#BC5CDR_WEIGHT = 'weights/bc5cdr_wt.pt'
#BIONLP13CG_WEIGHT = 'weights/bionlp13cg_wt.pt'
BERT_WEIGHTS = 'weights/'+BERT_MODEL+'/pytorch_weight'




