import torch
import torch.nn as nn
import numpy


class GRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, emb_size):
        super(GRU, self).__init__()
        self.hidden_size = hidden_size
        self.encoder = nn.Embedding(input_size,emb_size)
        self.gru = nn.GRU(emb_size,hidden_size)
        self.linear = nn.Linear(hidden_size,output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        
        input = self.encoder(input.long())
        print(input.size(),1123323)
        output,hidden = self.gru(input,hidden)
        output = self.linear(output[-1])
        output = self.softmax(output)
        
        return output, hidden
        
    def initHidden(self):
        return torch.zeros([1, self.hidden_size])


out_model_filename = './output/model_adult_gru_128.pt'
model = torch.load(out_model_filename)
print(model.encoder.weight.data)




