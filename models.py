import torch
import torch.optim as optim
from torch import nn
import hoge
import language
import configparser
import glob
import random
import joblib
import matplotlib.pyplot as plt
from scipy.signal import istft
import soundfile as sf

ja = language.Ja()
ini = configparser.ConfigParser()
ini.read("./config.ini")

class EncoderCBHG(nn.Module):
    def __init__(self):
        super(EncoderCBHG, self).__init__()
        encoder_filter = int(ini["modelparameter"]["encoder_filter"])
        encoder_filter_set = list(range(1, encoder_filter+1))

        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([nn.Conv1d(128, 128, filter_size, stride=1, padding=filter_size//2)\
                                    for filter_size in encoder_filter_set])
        self.pool = nn.MaxPool1d(2, stride=1, padding=0)
        self.conv = nn.Conv1d(2048, 128, 3, stride=1, padding=0)
        self.highway = Highway_net(128, 4, self.relu)
        self.gru = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(128)

    def forward(self, x):
        stacks = torch.Tensor()
        x = x.permute(0, 2, 1)
        input_origin = x
        T = x.shape[-1]
        for i in range(len(self.convs)):
            tmp = self.relu(self.convs[i](x))
            tmp = self.batchnorm(tmp)[:, :, :T]
            if i == 0:
                stacks = tmp
            else:
                stacks = torch.cat((stacks, tmp), 1)
        x = stacks
        x = self.pool(x)[:, :, :T]
        x = self.relu(self.conv(x))
        x = self.batchnorm(x)
        x += self.relu(input_origin) # residual
        x = self.highway(x.permute(0, 2, 1))
        x, h = self.gru(x)
        return x


class DecoderCBHG(nn.Module):
    def __init__(self):
        super(DecoderCBHG, self).__init__()
        decoder_filter = int(ini["modelparameter"]["decoder_filter"])
        decoder_filter_set = list(range(1, decoder_filter+1))

        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([nn.Conv1d(256, 128, filter_size, stride=1, padding=filter_size//2)\
                                    for filter_size in decoder_filter_set])
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.conv1 = nn.Conv1d(1024, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 128, 3, stride=1, padding=1)

        self.highway = Highway_net(128, 4, self.relu)
        self.gru = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True) # ワンチャン自分でgruの実装
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        stacks = torch.Tensor()
        x = x.permute(0, 2, 1)
        input_origin = x
        T = x.shape[-1]
        for i in range(len(self.convs)):
            tmp = self.relu(self.convs[i](x))
            tmp = self.batchnorm1(tmp)[:, :, :T]
            if i == 0:
                stacks = tmp
            else:
                stacks = torch.cat((stacks, tmp), 1)
        x = stacks
        x = self.pool(x)[:, :, :T]
        x = self.relu(self.conv1(x))
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = self.batchnorm1(x)
        x += self.relu(input_origin) # residual
        x = self.highway(x.permute(0, 2, 1))
        x, h = self.gru(x)
        return x

class Highway_net(nn.Module):
    def __init__(self, size, num_layers, f):
        super(Highway_net, self).__init__()
        self.num_layers = num_layers
        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])
        self.f = f
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        for layer in range(self.num_layers):
            gate = self.sigmoid(self.gate[layer](x))
            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)
            x = gate * nonlinear + (1 - gate) * linear
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(256, 256)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=2)

    def forward(self, h, d):
        d = d.expand(-1, h.shape[1], -1) # batch_size*1*128 =>
        u = self.tanh(self.w1(h)+self.w2(d))
        a = self.softmax(u)
        d = torch.sum(torch.mul(a, h), 1)
        return d

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.cbhg = EncoderCBHG()
        self.emb = nn.Embedding(len(ja.get_hiralist)+1, 256, padding_idx = len(ja.get_hiralist))
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, labels):
        x = self.emb(labels)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.cbhg(x)
        return x


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.cbhg = DecoderCBHG()
        self.emb = nn.Embedding(len(ja.get_hiralist), 256, padding_idx=-1)
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.rnn = nn.GRU(256, 256, 1, batch_first=True)
        self.attention = Attention()
        self.attention_rnn = nn.GRU(256, 256, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.r = int(ini["modelparameter"]["r"])
        self.batch_size = int(ini["hyperparameter"]["batch_size"])
        self.shrink = nn.Linear(256, 128)
        self.expand = nn.Linear(256, 2048)

    def forward(self, representation, spectrogram_len):
        x = hoge.try_gpu(torch.zeros(self.batch_size, 1, 256)) # <GO>
        x, attention_h = self.attention_rnn(x)
        a = self.attention(representation, x)
        decoder_h = a.view(1, self.batch_size, 256)
        # try 1layer
        x, decoder_h = self.rnn(x, decoder_h)
        y = x
        for _ in range(spectrogram_len-1):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            a = self.shrink(a).view(self.batch_size, 1, 128)
            x = torch.cat((x, a), 2)

            x, attention_h = self.attention_rnn(x)
            a = self.attention(representation, x)
            x, decoder_h = self.rnn(x, decoder_h)
            y = torch.cat((y, x), 1)
        y = self.cbhg(y)
        y = self.expand(y)
        return y

class Tacotron():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder = hoge.try_gpu(self.encoder)
        self.decoder = hoge.try_gpu(self.decoder)

        self.d_opt = optim.Adam(self.decoder.parameters())
        self.e_opt = optim.Adam(self.encoder.parameters())

        self.min_valid_loss = 10000000

        self.loss = nn.L1Loss()

        self.train_loss_transition = []
        self.valid_loss_transition = []

    def train(self, epoch):
        print("epoch[%d/%s]:"%(epoch, ini["hyperparameter"]["epoch_num"]))
        dirs = glob.glob(ini["directory"]["dataset"]+"/spectrogram/*")
        data_num = len(dirs)
        mini_batch_num = int(ini["hyperparameter"]["batch_size"])
        valid_rate = float(ini["hyperparameter"]["valid_rate"])
        display_step = 1000
        display_step_times = 1
        train_iter_max = (data_num/mini_batch_num) * (1-valid_rate)
        train_loss_total = 0
        valid_loss_total = 0

        # args of minibatch
        tmp = list(range(len(dirs)))
        random.shuffle(tmp)
        split_num = len(tmp)//mini_batch_num
        tmp = [tmp[(i-1)*mini_batch_num:i*mini_batch_num]\
            for i in range(1, len(dirs)//mini_batch_num+1)]
        for repeat, args in enumerate(tmp):
            if repeat < train_iter_max:
                if len(args) != mini_batch_num:
                    break
                if repeat%10==0:
                    print("      train iter[%d/%d]"%(repeat, train_iter_max-1))
                self.encoder.train()
                self.decoder.train()
                self.e_opt.zero_grad()
                self.d_opt.zero_grad()

                labels, datas = self.get_minibatch(args)
                labels = hoge.try_gpu(labels)
                datas = hoge.try_gpu(datas)
                representation = self.encoder(labels)
                predicted = self.decoder(representation, datas.shape[1])

                loss = self.loss(datas, predicted)
                loss.backward()

                self.d_opt.step()
                self.e_opt.step()
                train_loss_total += loss.item()
                if repeat%10==0:
                    print("             loss: %.3f"%(loss.item()*1000))

            else:
                if repeat%10==0:
                    print("      valid iter[%d/%d]"%(repeat-train_iter_max+1, int(data_num//mini_batch_num*valid_rate+1)))
                self.encoder.eval()
                self.decoder.eval()

                labels, datas = self.get_minibatch(args)
                labels = hoge.try_gpu(labels)
                datas = hoge.try_gpu(datas)
                representation = self.encoder(labels)
                predicted = self.decoder(representation, datas.shape[1])

                loss = self.loss(datas, predicted)
                valid_loss_total += loss.item()

        self.train_loss_transition.append(train_loss_total)
        self.valid_loss_transition.append(valid_loss_total)

        if self.min_valid_loss>valid_loss_total:
            self.min_valid_loss = valid_loss_total
            torch.save(self.decoder.state_dict(), "param/dweight")
            torch.save(self.encoder.state_dict(), "param/eweight")

    def evaluate(self, epoch):
        dirs = glob.glob(ini["directory"]["dataset"]+"/spectrogram/*")
        data_num = len(dirs)
        mini_batch_num = int(ini["hyperparameter"]["batch_size"])
        valid_rate = float(ini["hyperparameter"]["valid_rate"])
        sampling_rate = int(ini["signal"]["samplingrate"])
        display_step = 1000
        display_step_times = 1
        train_loss_total = 0
        valid_loss_total = 0

        # args of minibatch
        tmp = list(range(len(dirs)))
        sample = random.sample(tmp, mini_batch_num)

        self.encoder.eval()
        self.decoder.eval()

        labels, datas = self.get_minibatch(sample)
        labels = hoge.try_gpu(labels)
        datas = hoge.try_gpu(datas)

        representation = self.encoder(labels)
        predicted = self.decoder(representation, datas.shape[1])

        for i, (correct_arg, predict) in enumerate(zip(sample, predicted)):
            if i==3:
                break
            correct = joblib.load(ini["directory"]["dataset"]+"/waveform_data/%d"%(correct_arg))
            plt.figure()
            plt.plot(range(len(correct)), correct)
            plt.savefig("train_result/%d_%d_correct.png"%(epoch, correct_arg))
            plt.close()

            predict = predict.cpu().detach().numpy()
            predict = istft(predict)

            plt.figure()
            plt.plot(predict[0], predict[1])
            plt.savefig("train_result/%d_%d_predict.png"%(epoch, correct_arg))
            plt.close()

            sf.write('train_result/%d_%d_predict.wav'%(epoch, correct_arg), predict[1], sampling_rate)

    def get_minibatch(self, args):
        for i, arg in enumerate(args):
            data = joblib.load(ini["directory"]["dataset"]+"/spectrogram/%d"%(i))
            data = data.view([1]+list(data.shape))
            label = joblib.load(ini["directory"]["dataset"]+"/label_data/%d"%(i)).view(1, -1)
            if i == 0:
                datas = data
                labels = label
            else:
                datas = torch.cat((datas, data), 0)
                labels = torch.cat((labels, label), 0)
        return labels, datas

    def output(self, epoch):
        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(range(epoch), self.train_loss_transition, label = "train loss")
        plt.plot(range(epoch), self.valid_loss_transition, label = "valid loss")
        plt.legend()
        plt.savefig("train_result/loss_transition.png")
        plt.close()
