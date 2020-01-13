import torch
from torch import nn
import config
import hoge

class CBHG(nn.Module):
    def __init__():
        self.relu = nn.ReLU()
        # errorが出たらmodule dictに変更する予定
        # inputなどを加味して後で全てのconvにpoolを追加
        self.convs = nn.ModuleList([nn.Conv1d(1, 1, fileter_size, stride=1, padding=0)\
                                    for filter_size in filter_set])
        self.pool = nn.Maxpool1d(2, stride=1, padding=0)
        self.conv = nn.Conv1d(1, 1, 3, stride=1, padding=0)
        self.highway = highway_net(128, 4, self.relu)
        self.gru = nn.GRU(128, hidden_size, num_layers, batch_first=True, bidirectional=True) # ワンチャン自分でgruの実装


class highway_net(nn.Module):
    def __init__(self, size, num_layers, f):
        super(highway_net, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

class Encoder(nn.Module):
    def __init__():
        self.cbhg = CBHG()
        self.emb = nn.Embedding(, 256)
        self.fc1 = nn.Linear(256, 256)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

class Decoder(nn.Module):
    def __init__():
        self.cbhg = CBHG()

class tactron():
    def __init__():
        encoder = self.Encoder()
        decoder = self.Decoder()
