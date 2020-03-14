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
import numpy as np
import time
from torch.autograd import Variable

ja = language.Ja()
ini = configparser.ConfigParser()
ini.read("./config.ini")
dictionary = joblib.load("param/dictionary")

class EncoderCBHG(nn.Module):
    def __init__(self):
        super(EncoderCBHG, self).__init__()
        encoder_filter = int(ini["modelparameter"]["encoder_filter"])
        encoder_filter_set = list(range(1, encoder_filter+1))

        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([nn.Conv1d(128, 128, filter_size, stride=1, padding=filter_size//2)\
                                    for filter_size in encoder_filter_set])
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.conv1 = nn.Conv1d(2048, 128, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(128, 128, 3, stride=1, padding=1)
        self.highway = Highway_net(128, 4, self.relu)
        self.gru = nn.GRU(128, 128, 1, batch_first=True, bidirectional=True)
        self.batchnorm = nn.BatchNorm1d(128)

    def forward(self, x):
        stacks = torch.Tensor()
        x = x.permute(0, 2, 1)
        input_origin = x
        T = x.shape[-1]
        for i in range(len(self.convs)):
            tmp = self.convs[i](x)
            tmp = self.relu(tmp)
            tmp = self.batchnorm(tmp)
            tmp = tmp[:, :, :T]
            if i == 0:
                stacks = tmp
            else:
                stacks = torch.cat((stacks, tmp), 1)
        x = stacks
        del stacks
        x = self.pool(x)[:, :, :T]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm(x)
        x = self.conv2(x)
        x = self.batchnorm(x)
        x = x + input_origin # residual
        x = self.highway(x.permute(0, 2, 1))
        x, h = self.gru(x)
        return x


class DecoderCBHG(nn.Module):
    def __init__(self):
        super(DecoderCBHG, self).__init__()
        self.batch_size = int(ini["hyperparameter"]["batch_size"])
        self.r = int(ini["modelparameter"]["r"])
        self.melsize = int(ini["signal"]["band"])
        decoder_filter = int(ini["modelparameter"]["decoder_filter"])
        decoder_filter_set = list(range(1, decoder_filter+1))

        self.relu = nn.ReLU()
        self.convs = nn.ModuleList([nn.Conv1d(80, 128, filter_size, stride=1, padding=filter_size//2)\
                                    for filter_size in decoder_filter_set])
        self.pool = nn.MaxPool1d(2, stride=1, padding=1)
        self.conv1 = nn.Conv1d(1024, 256, 3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 80, 3, stride=1, padding=1)

        self.highway = Highway_net(80, 4, self.relu)
        self.gru = nn.GRU(80, 80, 1, batch_first=True, bidirectional=True)
        self.batchnorm1 = nn.BatchNorm1d(128)
        self.batchnorm2 = nn.BatchNorm1d(256)
        self.batchnorm3 = nn.BatchNorm1d(80)

    def forward(self, x):
        stacks = torch.Tensor()
        x = x.permute(0, 2, 1)
        input_origin = x
        T = x.shape[-1]
        for i in range(len(self.convs)):
            tmp = self.convs[i](x)
            tmp = self.relu(tmp)
            tmp = self.batchnorm1(tmp)
            tmp = tmp[:, :, :T]
            if i == 0:
                stacks = tmp
            else:
                stacks = torch.cat((stacks, tmp), 1)
        x = stacks
        del stacks
        x = self.pool(x)[:, :, :T]
        x = self.conv1(x)
        x = self.relu(x)
        x = self.batchnorm2(x)
        x = self.conv2(x)
        x = self.batchnorm3(x)
        x = x + input_origin # residual
        x = self.highway(x.permute(0, 2, 1))
        x, h = self.gru(x)
        return x

class Prenet(nn.Module):
    def __init__(self, dim):
        super(Prenet, self).__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
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
            x = gate * nonlinear + (1 - gate) * x
            #x = gate * nonlinear + (1 - gate) * linear
        return x

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        self.w1 = nn.Linear(256, 256)
        self.w2 = nn.Linear(256, 256)
        self.v = nn.Linear(256, 1)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, h, d):
        #d = d.expand(-1, h.shape[1], -1) # batch_size*1*128 =>
        #u = self.v(self.tanh(self.w1(h)+self.w2(d)))
        u = self.v(self.tanh(self.w1(h)+self.w2(d)))
        u = u.squeeze(-1)
        a = self.softmax(u)
        attention = torch.bmm(a.unsqueeze(1), h)
        return attention

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        emb = int(ini["lang"]["emb"])
        self.prenet = Prenet(emb)
        self.cbhg = EncoderCBHG()
        self.emb = nn.Embedding(len(dictionary)+1, emb, padding_idx = len(dictionary))

    def forward(self, labels):
        x = self.emb(labels)
        x = self.prenet(x)
        x = self.cbhg(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.batch_size = int(ini["hyperparameter"]["batch_size"])
        self.r = int(ini["modelparameter"]["r"])
        self.melsize = int(ini["signal"]["band"])
        self.emb = int(ini["lang"]["emb"])

        self.prenet = Prenet(self.melsize)
        self.cbhg = DecoderCBHG()
        self.attention = Attention()
        self.attention_rnn = nn.GRU(256+128, 256, 1, batch_first=True)
        self.rnns = nn.ModuleList([nn.GRU(256, 256, batch_first=True) for _ in range(2)])
        self.shrink = nn.Linear(256+256, 256)
        #self.emb_shrink = nn.Linear(256, self.emb)
        self.mel = nn.Linear(256, self.melsize*self.r)
        self.linear = nn.Linear(self.melsize*2, 1025)
        self.mask = nn.Linear(self.melsize, 1)
        #self.mask = nn.Linear(256//self.r, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, representation, spectrogram_len, correct_mels=None):
        # first step
        x = Variable(hoge.try_gpu(torch.zeros(self.batch_size, 1, self.melsize))) # <GO>
        a = Variable(hoge.try_gpu(torch.zeros(self.batch_size, 1, 256)))
        decoder_h = Variable(hoge.try_gpu(torch.zeros(1, self.batch_size, 256)))
        if not correct_mels is None:
            tmps = hoge.try_gpu(torch.Tensor())
            for arg in range(self.r, correct_mels.shape[1], self.r):
                args = list(range(arg-self.r, arg))
                tmp = torch.flatten(correct_mels[:, args, :], start_dim=1)
                tmp = tmp.unsqueeze(1)
                tmps = torch.cat((tmps, tmp), 1)
            correct_mels = tmps

        x = self.prenet(x)
        x = torch.cat((x, a), 2)

        x, attention_h = self.attention_rnn(x)
        a = self.attention(representation, x)

        x = self.shrink(torch.cat((x, a), 2))
        #attention_h = x.permute(1, 0, 2)
        #attention_h = torch.cat((attention_h, a.permute(1, 0, 2)), 2)

        decoder_hs = []
        for i, rnn in enumerate(self.rnns):
            renew, decoder_h = rnn(x, decoder_h)
            x = x + renew
            decoder_hs.append(decoder_h)
        x = self.mel(x)
        y = x

        for i in range(int(np.floor((spectrogram_len-self.r)/self.r))):
        #for _ in range(spectrogram_len-1):
            #x = self.emb_shrink(x)
            if not correct_mels is None:
                x = correct_mels[:, i, :]
                x = x.unsqueeze(1)
            x = x[:, :, -1*self.melsize-1:-1]
            x = self.prenet(x)

            x = torch.cat((x, a), 2)
            #attention_h = torch.cat((attention_h, a.permute(1, 0, 2)), 2)
            x, attention_h = self.attention_rnn(x, attention_h)
            a = self.attention(representation, x)

            x = self.shrink(torch.cat((x, a), 2))
            attention_h = x.permute(1, 0, 2)

            for i, rnn in enumerate(self.rnns):
                renew, decoder_hs[i] = rnn(x, decoder_hs[i])
                x = x + renew
            x = self.mel(x)
            y = torch.cat((y, x), 1)
        #mel = self.relu(self.mel(y))
        #mel = self.tanh(self.mel(y))
        #tmp = y.view(y.shape[0], -1, 256//self.r)
        #mask = self.sigmoid(self.mask(tmp))
        #mel = self.mel(y)
        pred_mels = y
        mel = y.view(y.shape[0], -1, self.melsize)
        mask = self.sigmoid(self.mask(mel))
        y = self.cbhg(mel)
        linear = self.linear(y)
        #linear = self.relu(self.linear(y))
        #linear = self.tanh(self.linear(y))
        if not correct_mels is None:
            return mel, linear, mask, correct_mels
        else:
            return mel, linear, mask

class Tacotron():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

        self.encoder = hoge.try_gpu(self.encoder)
        self.decoder = hoge.try_gpu(self.decoder)

        self.d_opt = optim.Adam(self.decoder.parameters(), lr=2e-3)
        self.e_opt = optim.Adam(self.encoder.parameters(), lr=2e-3)

        step_first = int(ini["hyperparameter"]["opt_step"])*170
        self.d_scheduler = optim.lr_scheduler.MultiStepLR(self.d_opt,\
                milestones=[step_first, step_first*2, step_first*4, step_first*8], gamma=0.5)
        self.e_scheduler = optim.lr_scheduler.MultiStepLR(self.e_opt,\
                milestones=[step_first, step_first*2, step_first*4, step_first*8], gamma=0.5)

        self.min_valid_loss = 10000000

        self.loss = nn.L1Loss()
        #self.loss = nn.MSELoss()

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

        start_time = time.time()
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

                labels, linears, mels = self.get_minibatch(args)
                labels = Variable(hoge.try_gpu(labels))
                linears = Variable(hoge.try_gpu(linears))
                mels = Variable(hoge.try_gpu(mels))

                representation = self.encoder(labels)
                pred_mels, pred_linears, pred_mask, correct_mels = self.decoder(representation, linears.shape[1], mels)

                length = min(mels.shape[1], pred_mels.shape[1])

                # sum is not 0(=not 0 padding)
                mask = torch.sum(mels, 2)
                mask = [mask != 0][0].unsqueeze(-1)
                correct_mask = mask
                mask = mask.expand(-1, -1, 80)
                mel_loss = 0.5*self.loss(mels[:,:length,:], pred_mels[:,:length,:]) +\
                        0.5*self.loss(mels[:, :length, :], (pred_mels[:,:length,:]*mask[:,:length,:]))
                """
                mel_loss = self.loss(correct_mels[:,:length,:], pred_mels[:,:length,:])
                """

                # sum is not 0(=not 0 padding)
                mask = torch.sum(linears,2)
                mask = [mask != 0][0].unsqueeze(-1)
                mask = mask.expand(-1, -1, 1025)
                linear_loss = 0.5*self.loss(linears[:,:length,:], pred_linears[:,:length,:]) +\
                        0.5*self.loss(linears[:, :length, :], (pred_linears[:,:length,:]*mask[:,:length,:]))

                """
                correct_mask = np.array(correct_mask.cpu().detach().numpy(), dtype=np.float)
                correct_mask = hoge.try_gpu(torch.Tensor(correct_mask))
                mask_loss = self.loss(correct_mask[:,:length,:], pred_mask[:,:length,:])
                """
                """
                # sum is not 0(=not 0 padding)
                mask = torch.sum(mels, 2)
                mask = [mask != 0][0].unsqueeze(-1)
                mask = mask.cpu().detach().numpy()
                mask = hoge.try_gpu(torch.Tensor(np.array(mask, dtype=np.float)))
                mask = mask.expand(-1, -1, 80)
                mel_loss = 0.5*self.loss(mels[:, :length, :], (pred_mels[:,:length,:]*mask[:,:length,:]))

                # sum is not 0(=not 0 padding)
                mask = torch.sum(linears,2)
                mask = [mask != 0][0].unsqueeze(-1)
                mask = mask.cpu().detach().numpy()
                mask = hoge.try_gpu(torch.Tensor(np.array(mask, dtype=np.float)))
                correct_mask = mask
                mask = mask.expand(-1, -1, 1025)
                linear_loss = 0.5*self.loss(linears[:, :length, :], (pred_linears[:,:length,:]*mask[:,:length,:]))

                #correct_mask = np.array(correct_mask.cpu().detach().numpy(), dtype=np.float)
                #correct_mask = hoge.try_gpu(torch.Tensor(correct_mask))
                mask_loss = self.loss(correct_mask[:,:length,:], pred_mask[:,:length,:])
                #mel_loss = self.loss(mels[:,:length,:], pred_mels[:,:length,:])
                #linear_loss = self.loss(linears[:,:length,:], pred_linears[:,:length,:])
                #loss = linear_loss + mask_loss
                """
                loss = mel_loss + linear_loss
                loss.backward()

                self.d_opt.step()
                self.e_opt.step()
                train_loss_total = train_loss_total + float(loss.item())
                if repeat%100==0:
                    print("                loss: %.3f"%(loss.item()*1000))
                    print("            mel loss: %.3f"%(mel_loss.item()*1000))
                    print("         linear loss: %.3f"%(linear_loss.item()*1000))
                    #print("           mask loss: %.3f"%(mask_loss.item()*1000))
                    #self.evaluate(epoch*10000+repeat)
                self.d_scheduler.step()
                self.e_scheduler.step()
                del loss, linear_loss, mel_loss, representation,\
                        pred_mels, pred_linears, labels, linears, mels

            else:
                if repeat%10==0:
                    print("      valid iter[%d/%d]"%(repeat-train_iter_max+1, int(data_num//mini_batch_num*valid_rate+1)))
                self.encoder.eval()
                self.decoder.eval()
                self.e_opt.zero_grad()
                self.d_opt.zero_grad()

                labels, linears, mels = self.get_minibatch(args)
                labels = hoge.try_gpu(labels)
                linears = hoge.try_gpu(linears)
                mels = hoge.try_gpu(mels)
                representation = self.encoder(labels)
                pred_mels, pred_linears, _ = self.decoder(representation, linears.shape[1])

                length = min(mels.shape[1], pred_mels.shape[1])
                mel_loss = self.loss(mels[:,:length,:], pred_mels[:,:length,:])
                linear_loss = self.loss(linears[:,:length,:], pred_linears[:,:length,:])
                #mel_loss = self.loss(mels, pred_mels)
                #linear_loss = self.loss(linears, pred_linears)
                loss = mel_loss + linear_loss
                loss = loss.item()
                valid_loss_total = valid_loss_total + float(loss)
                del loss, linear_loss, mel_loss, representation,\
                        pred_mels, pred_linears, labels, linears, mels

        print("implemented time: %.3f"%(time.time()-start_time))
        self.train_loss_transition.append(train_loss_total/train_iter_max)
        self.valid_loss_transition.append(valid_loss_total/(repeat-train_iter_max))

        """
        if self.min_valid_loss>valid_loss_total:
            self.min_valid_loss = valid_loss_total
            torch.save(self.decoder.state_dict(), "param/dweight")
            torch.save(self.encoder.state_dict(), "param/eweight")
        """
        if self.min_valid_loss>train_loss_total:
            self.min_valid_loss = train_loss_total
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

        labels, linears, mels = self.get_minibatch(sample)
        labels = hoge.try_gpu(labels)
        linears = hoge.try_gpu(linears)
        mels = hoge.try_gpu(mels)

        representation = self.encoder(labels)
        pred_mels, pred_linears, _, _ = self.decoder(representation, linears.shape[1], mels)
        pred_mels2, pred_linears2, _ = self.decoder(representation, linears.shape[1])

        # visualization
        for i, correct_arg in enumerate(sample):
            if i==3:
                break
            # correct spectrogram
            correct = joblib.load(ini["directory"]["dataset"]+"/spectrogram/%d"%(correct_arg))
            plt.figure()
            plt.pcolormesh(correct)
            plt.colorbar()
            plt.savefig("train_result/%d_%d_correct_line.png"%(epoch, correct_arg))
            plt.close()

            # predicted spectrogram
            predict = pred_linears[i].cpu().detach().numpy()
            plt.figure()
            plt.pcolormesh(predict)
            plt.colorbar()
            plt.savefig("train_result/%d_%d_pred_line1.png"%(epoch, correct_arg))
            plt.close()

            # predicted spectrogram
            predict = pred_linears2[i].cpu().detach().numpy()
            plt.figure()
            plt.pcolormesh(predict)
            plt.colorbar()
            plt.savefig("train_result/%d_%d_pred_line2.png"%(epoch, correct_arg))
            plt.close()

            # predicted waveform
            predict = istft(predict)
            plt.figure()
            plt.plot(predict[0], predict[1])
            plt.savefig("train_result/%d_%d_predict_wave.png"%(epoch, correct_arg))
            plt.close()

            # wav data
            sf.write("train_result/%d_%d_predict_wave.wav"%(epoch, correct_arg), predict[1], sampling_rate)

            # correct melspectrogram
            correct = joblib.load(ini["directory"]["dataset"]+"/melspectrogram/%d"%(correct_arg))
            plt.figure()
            plt.pcolormesh(correct)
            plt.colorbar()
            plt.savefig("train_result/%d_%d_correct_mel.png"%(epoch, correct_arg))
            plt.close()

            # predicted melspectrogram
            predict = pred_mels[i].cpu().detach().numpy()
            plt.figure()
            plt.pcolormesh(predict)
            plt.colorbar()
            plt.savefig("train_result/%d_%d_pred_mel1.png"%(epoch, correct_arg))
            plt.close()

            # predicted melspectrogram
            predict = pred_mels2[i].cpu().detach().numpy()
            plt.figure()
            plt.pcolormesh(predict)
            plt.colorbar()
            plt.savefig("train_result/%d_%d_pred_mel2.png"%(epoch, correct_arg))
            plt.close()

    def get_minibatch(self, args):
        for i, arg in enumerate(args):
            spectrogram = joblib.load(ini["directory"]["dataset"]+"/spectrogram/%d"%(i))
            spectrogram = spectrogram.view([1]+list(spectrogram.shape))
            mel = joblib.load(ini["directory"]["dataset"]+"/melspectrogram/%d"%(i))
            mel = mel.view([1]+list(mel.shape))
            label = joblib.load(ini["directory"]["dataset"]+"/label_data/%d"%(i)).view(1, -1)
            if i == 0:
                spectrograms = spectrogram
                mels = mel
                labels = label
            else:
                spectrograms = torch.cat((spectrograms, spectrogram), 0)
                mels = torch.cat((mels, mel), 0)
                labels = torch.cat((labels, label), 0)
        return labels, spectrograms, mels

    def output(self, epoch):
        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(range(epoch), self.train_loss_transition, label = "train loss")
        plt.plot(range(epoch), self.valid_loss_transition, label = "valid loss")
        plt.legend()
        plt.savefig("train_result/loss_transition.png")
        plt.close()

        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(range(epoch), self.train_loss_transition, label = "train loss")
        plt.legend()
        plt.savefig("train_result/train_loss_transition.png")
        plt.close()

        plt.figure()
        plt.xlabel("epochs")
        plt.ylabel("loss")
        plt.plot(range(epoch), self.valid_loss_transition, label = "valid loss")
        plt.legend()
        plt.savefig("train_result/valid_loss_transition.png")
        plt.close()
