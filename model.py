import torch
import torch.nn as nn
from torch.autograd import Variable


class lstm_activity(nn.Module):

    def __init__(self, x_dim, h_dim, batch_size, n_layers, output_dim):

        super(lstm_activity, self).__init__()

        self.x_dim = x_dim
        self.h_dim = h_dim
        self.n_layers = n_layers
        self.output_dim = output_dim
        self.lstm = nn.LSTM(input_size=x_dim, hidden_size=h_dim,
                            num_layers=n_layers, dropout=0.25, bidirectional=True)
        self.batch_size = batch_size
        self.dropout = nn.Dropout(0.25)
        self.fc = nn.Sequential(
            nn.Linear(h_dim * 2, output_dim),
        )
        self.hidden = self.init_hidden()

    def init_hidden(self):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.h_dim)),
                    Variable(torch.zeros(self.n_layers * 2, self.batch_size, self.h_dim)))

    def init_hidden_pred(self, len_pred):
        if torch.cuda.is_available():
            return (Variable(torch.zeros(self.n_layers * 2, len_pred, self.h_dim)).cuda(),
                    Variable(torch.zeros(self.n_layers * 2, len_pred, self.h_dim)).cuda())
        else:
            return (Variable(torch.zeros(self.n_layers * 2, len_pred, self.h_dim)),
                    Variable(torch.zeros(self.n_layers * 2, len_pred, self.h_dim)))

    def forward(self, x):

        lstm_out, self.hidden = self.lstm(x, self.hidden)
        #last_out = lstm_out[-1].view(-1, self.h_dim)
        lstm_out = self.dropout(lstm_out)
        result = self.fc(lstm_out[-1].view(-1, self.h_dim * 2))
        #last_out = self.dropout(last_out)
        #result = self.fc(last_out)
        return result
