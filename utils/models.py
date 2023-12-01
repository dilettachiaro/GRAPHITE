import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.parameter import Parameter
import math

# GRU
class GRU_model(nn.Module):
    def __init__(self, step, is_cuda):
        super(GRU_model, self).__init__()
        self.is_cuda = is_cuda
        self.rnn_like = nn.GRU(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0,
            bidirectional=False)

        self.linear1 = nn.Linear(64, step, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = x.permute(1, 0, 2)

        if self.is_cuda:
            h_0 = Variable(torch.zeros(2, x.shape[1], 64).cuda())
        else:
            h_0 = Variable(torch.zeros(2, x.shape[1], 64))
        
        x, _ = self.rnn_like(x, h_0)
        x = x.permute(1, 0, 2) 
        x = self.linear1(x[:, -1, :])
        x = self.tanh(x)

        return x


# LSTM
class LSTM_model(nn.Module):
    def __init__(self, step, is_cuda):
        super(LSTM_model, self).__init__()
        self.is_cuda = is_cuda
        self.rnn_like = nn.LSTM(
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0,
            bidirectional=False)

        self.linear1 = nn.Linear(64, step, bias=False)
        self.tanh = nn.Tanh()


    def forward(self, x):
        x = x.permute(1, 0, 2)

        if self.is_cuda:
            h_0 = Variable(torch.zeros(2, x.shape[1], 64).cuda())
            c_0 = Variable(torch.zeros(2, x.shape[1], 64).cuda())
        else:
            h_0 = Variable(torch.zeros(2, x.shape[1], 64))
            c_0 = Variable(torch.zeros(2, x.shape[1], 64))

        x, _ = self.rnn_like(x, (h_0, c_0))
        x = x.permute(1, 0, 2)
        x = self.linear1(x[:, -1, :])
        x = self.tanh(x)

        return x


# GCN
class GraphConvolution(nn.Module):
    def __init__(self, step, is_cuda):
        super(GraphConvolution, self).__init__()
        self.is_cuda = is_cuda
        self.weight = Parameter(torch.FloatTensor(1, 1)).cuda()
        self.reset_parameters()
        self.fc = nn.Linear(28, step, bias=False)
        self.tanh = nn.Tanh()

    def reset_parameters(self):
        if self.is_cuda:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv).cuda()

        else:
            stdv = 1. / math.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)



    def forward(self, x_f, kg_1, kg_2):
        support1 = torch.matmul(kg_1, x_f).cuda()
        support2 = torch.matmul(kg_2, x_f).cuda()
        x = support1 + support2
        x = x.view(-1, 28)
        x = self.fc(x)
        x = self.tanh(x)
        x = x.unsqueeze(2)

        return x


# MLP
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(4, 32, bias=False)
        self.tanh = nn.Tanh()
        self.linear2 = nn.Linear(32, 3, bias=False)

    def forward(self, x):
        x = self.linear1(x)
        x = self.tanh(x)
        x = self.linear2(x)

        return x
