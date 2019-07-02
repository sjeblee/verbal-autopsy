import dataloaders_experimental as dte 
import models_experimental as mdl 
import train_experimental as tr
import sys 

PATH_TO_DATA = './hindi_all_adult.xml'

# load and prepare data 
full_data = dte.load_data(PATH_TO_DATA)
prepared_data = dte.prepare_data(full_data)
print(len(prepared_data))

# create the embedding matrix
deva_index, embed_mat = dte.word_vectorize_data(prepared_data)
print(embed_mat[:3,:])

