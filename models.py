import torch
from torch import nn
import hoge
import language
import configparser

ja = language.Ja()
ini = configparser.ConfigParser()
ini.read("./config.ini")

class EncoderCBHG(nn.Module):
    def __init__(self):
        super(EncoderCBHG, self).__init__()
        encoder_filter = int(ini["modelparameter"]["encoder_filter"])
        encoder_filter_set = list(range(1, encoder_filter+1))

        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([nn.Conv1d(1, 1, filter_size, stride=1, padding=0)\
                                    for filter_size in encoder_filter_set])
        self.pool = nn.MaxPool1d(2, stride=1, padding=0)
        self.conv = nn.Conv1d(1, 1, 3, stride=1, padding=0)
        self.highway = highway_net(128, 4, self.relu)
        self.gru = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True) # ワンチャン自分でgruの実装

class DecoderCBHG(nn.Module):
    def __init__(self):
        super(DecoderCBHG, self).__init__()
        decoder_filter = int(ini["modelparameter"]["decoder_filter"])
        decoder_filter_set = list(range(1, decoder_filter+1))

        self.relu = nn.ReLU()
        # errorが出たらmodule dictに変更する予定
        # inputなどを加味して後で全てのconvにpoolを追加
        self.convs = nn.ModuleList([nn.Conv1d(1, 1, filter_size, stride=1, padding=0)\
                                    for filter_size in decoder_filter_set])
        # 256, 80になるように調整
        self.conv1 = nn.Conv1d(1, 1, 3, stride=1, padding=0)
        self.conv2 = nn.Conv1d(1, 1, 3, stride=1, padding=0)

        self.highway = highway_net(128, 4, self.relu)
        self.gru = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True) # ワンチャン自分でgruの実装

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
    def __init__(self):
        super(Encoder, self).__init__()
        self.cbhg = EncoderCBHG()
        self.emb = nn.Embedding(len(ja.get_hiralist), 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

# residual機能を使う
# all-zeroを最初のフレームとして用いる
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cbhg = DecoderCBHG()
        self.emb = nn.Embedding(len(ja.get_hiralist), 256)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

# griffin limは振幅スペクトルから位相スペクトルを再現する手法
# griffin limは古いものなので, waveRNNなどを使ったほうがいい結果になることもありけり。らしい
# 参照 https://www.jstage.jst.go.jp/article/jasj/72/12/72_764/_pdf

# https://qiita.com/KSRG_Miyabi/items/2a3b5bdca464ec1154d7
# スペクトル絶対値Aを適当な位相Nで初期化しXを作る。　( X = A * N )
# ① x = IFT(X)
# ② X = FT(x)
# ③ X = A * X / |X|
# ①～③を適当な数(50~100回？)だけ繰り返す。
# paper曰く今回は50(なお30でも充分らしい)

# わんちゃんこれか? https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.istft.html
class Tacotron():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()
