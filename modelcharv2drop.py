#Georges Belanger Albarran - Charachter language model

#Based on the following blogs:
#Sean Robertson Practical Pytorch: Classifying names with a charachter level RNN https://github.com/spro/practical-pytorch/blob/master/char-rnn-classification/char-rnn-classification.ipynb
#Soumith Chintala Word Language Model: https://github.com/pytorch/examples/tree/master/word_language_model
#--------------------------------------
import torch.nn as nn
from torch.autograd import Variable

class RNNModel(nn.Module):

    def __init__(self, ntoken, ninp, nhid, nlayers):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(0.5)

        self.encoder = nn.Embedding(ntoken, ninp)
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=0.5)
        
        #self.rnn = nn.LSTM(ntoken, nhid, nlayers)

        self.decoder = nn.Linear(nhid, ntoken)

        self.init_weights()

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        
        self.encoder.weight.data.uniform_(-initrange, initrange)
        
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        
        emb = self.encoder(input)
        output, hidden = self.rnn(emb, hidden)
        
        #output, hidden = self.rnn(input, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        return (Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()), 
            Variable(weight.new(self.nlayers, bsz, self.nhid).zero_()))


