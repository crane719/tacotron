import soundfile as sf
import numpy as np
import hoge
import configparser
import joblib
import language
from scipy import signal
from scipy.fftpack import fft, fftshift
import math
import torch
from torch.utils.data import TensorDataset, DataLoader
import librosa
import scipy

class Preprocess():
    def __init__(self):
        # read config
        ini = configparser.ConfigParser()
        ini.read("./config.ini")

        # create required directory
        required = [ini["directory"]["dataset"],
                    ini["directory"]["dataset"]+"/label_data",
                    ini["directory"]["dataset"]+"/spectrogram",
                    ini["directory"]["dataset"]+"/melspectrogram",
                    ini["directory"]["dataset"]+"/waveform_data"]
        boolean = [hoge.is_dir_existed(dir) for dir in required]
        for bool, directory in zip(boolean, required):
            if bool:
                hoge.recreate_dir([directory])
            else:
                hoge.make_dir([directory])
        print("\n")

        print("Start preprocess:")

        # signal discription
        print("     NLP")
        ja = language.Ja()
        f = open(ini["directory"]["raw_dataconf"], "r")
        raw_dataconf = f.read()
        raw_dataconf = list(reversed(raw_dataconf.split("\n")))
        labels = []
        i = 0
        max_len = 0
        data_len = len(raw_dataconf)
        dictionary = []
        while len(raw_dataconf):
            if i%1000==0:
                print("     [%d/%d]"%(i, data_len))
            col = raw_dataconf.pop()

            if len(col) == 0:
                break
            """
            text = col.split("|")[1]
            label = ja.kana2label(text)
            """
            text = col.split("|")[1]
            #print(col.split("|")[0], i)
            #text = text.replace("\n", "")
            text = ja.del_symbol(text)
            text = ja.split2list(text)[:-1]

            dictionary.extend(text)
            dictionary = list(np.unique(dictionary))

            if len(text) > max_len:
                max_len = len(text)
            #joblib.dump(torch.LongTensor(text), ini["directory"]["dataset"]+"/label_data/%d"%(i), compress=3)
            joblib.dump(text, ini["directory"]["dataset"]+"/label_data/%d"%(i), compress=3)
            i += 1

        dictionary = list(sorted(dictionary))
        joblib.dump(dictionary, "param/dictionary", compress=3)

        # padding and labeling
        print("     padding and labeling")
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/label_data/*")
        dirs = list(sorted(dirs))
        for filenum,directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            text = joblib.load(directory)
            label = [np.where(np.array(dictionary) == word)[0][0] for word in text]
            label = np.array(label)
            diff = max_len - label.shape[0]
            label = np.concatenate((label, np.ones((diff))*len(dictionary)), 0)
            joblib.dump(torch.LongTensor(label), directory, compress=3)#ini["directory"]["dataset"]+"/label_data/%d"%(i), compress=3)

        print("     signal processing")
        filedirs = hoge.get_filedirs(ini["directory"]["raw_data"]+"/*")
        max_len = 0
        max_mel_len = 0

        #ave = []
        #mel_ave = []
        ave = 0
        mel_ave = 0
        elements = 0
        mel_elements = 0
        for filenum, filedir in enumerate(filedirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(filedirs)))
            data, rate = sf.read(filedir)
            filenum = filedir.split("_")[1]
            filenum = filenum.split(".wav")[0]
            filenum = int(filenum)

            # resampling
            """
            wanted_rate_bias = int(ini["signal"]["samplingrate"])/rate
            wanted_data_len = int(len(data) * wanted_rate_bias)
            data = signal.resample(data, wanted_data_len)
            """
            joblib.dump(data, ini["directory"]["dataset"]+"/waveform_data/%d"%(filenum), compress=3)

            window_shift = int(ini["signal"]["window_shift"])
            window_size = int(ini["signal"]["window_size"])
            fft_point = int(ini["signal"]["fft_point"])
            fs = int(ini["signal"]["samplingrate"])
            em = float(ini["signal"]["em"])
            th = float(ini["signal"]["no_voice_db_th"])

            if ini["signal"]["window"] == "Hann": window = signal.hann(window_size)

            # pre-empahasis
            renew_data = signal.lfilter([1.0, -em], 1, data)
            # stft
            spectrogram = librosa.core.stft(renew_data, n_fft=fft_point, hop_length=window_shift, win_length=window_size, window='hann')
            spectrogram = np.abs(spectrogram)
            spectrogram = np.log(spectrogram+1e-3)
            spectrogram = spectrogram.T
            # no voice cut
            """
            del_args = np.where(np.max(spectrogram, axis=1) < th)[0]
            spectrogram = np.delete(spectrogram, del_args, axis=0)
            """
            joblib.dump(spectrogram, ini["directory"]["dataset"]+"/spectrogram/%d"%(filenum), compress=3)

            # renew max len
            if spectrogram.shape[0] > max_len:
                max_len = spectrogram.shape[0]
            ave += np.sum(spectrogram.reshape(-1))
            elements += len(spectrogram.reshape(-1))

            # melspectrogram
            mel_spectrogram = librosa.feature.melspectrogram(renew_data,
                                                             S=spectrogram.T,
                                                             n_mels=int(ini["signal"]["band"]))
            mel_spectrogram = mel_spectrogram.T
            joblib.dump(mel_spectrogram, ini["directory"]["dataset"]+"/melspectrogram/%d"%(filenum), compress=3)
            if mel_spectrogram.shape[0] > max_mel_len:
                max_mel_len = mel_spectrogram.shape[0]
            mel_ave += np.sum(mel_spectrogram.reshape(-1))
            mel_elements += len(mel_spectrogram.reshape(-1))

        print("     padding...")
        var = 0
        mel_var = 0
        ave /= elements
        mel_ave /= mel_elements
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        for filenum, directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            spectrogram = joblib.load(directory)
            var += np.sum((spectrogram-ave)**2)
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        # melspectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/melspectrogram/*")
        for filenum, directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            spectrogram = joblib.load(directory)
            mel_var += np.sum((spectrogram-mel_ave)**2)
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        print("     normalization...")
        var /= elements
        mel_var /= mel_elements
        var = np.sqrt(var)
        mel_var = np.sqrt(mel_var)
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        for filenum, directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            spectrogram = joblib.load(directory)
            spectrogram = (spectrogram - ave)/var
            diff = max_len - spectrogram.shape[0]
            spectrogram = np.concatenate((spectrogram, np.zeros((diff, spectrogram.shape[1]))), 0)
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        # melspectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/melspectrogram/*")
        for filenum, directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            spectrogram = joblib.load(directory)
            spectrogram = (spectrogram - mel_ave)/mel_var
            diff = max_mel_len - spectrogram.shape[0]
            spectrogram = np.concatenate((spectrogram, np.zeros((diff, spectrogram.shape[1]))), 0)
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)
