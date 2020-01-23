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
import pickle
from torch.utils.data import TensorDataset, DataLoader

class Preprocess():
    def __init__(self):
        # read config
        ini = configparser.ConfigParser()
        ini.read("./config.ini")

        # create required directory
        required = [ini["directory"]["dataset"],
                    ini["directory"]["dataset"]+"/label_data",
                    ini["directory"]["dataset"]+"/spectrogram",
                    ini["directory"]["dataset"]+"/waveform_data"]
        boolean = [hoge.is_dir_existed(dir) for dir in required]
        for tf, directory in zip(boolean, required):
            if tf:
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
        data_len = len(raw_dataconf)
        while len(raw_dataconf):
            if i%100==0:
                print("     [%d/%d]"%(i, data_len))
            col = raw_dataconf.pop()

            if len(col) == 0:
                break
            text = col.split("|")[1]
            label = ja.kana2label(text)
            joblib.dump(label, ini["directory"]["dataset"]+"/label_data/%d"%(i), compress=3)
            i += 1

        print("     signal processing")
        filedirs = hoge.get_filedirs(ini["directory"]["raw_data"]+"/*")
        max_len = 0
        for filenum, filedir in enumerate(filedirs):
            if (filenum+1)%100==0:
                print("     [%d/%d]"%(filenum, len(filedirs)))
            data, rate = sf.read(filedir)

            # resampling
            wanted_rate_bias = int(ini["signal"]["samplingrate"])/rate
            wanted_data_len = int(len(data) * wanted_rate_bias)
            data = signal.resample(data, wanted_data_len)
            joblib.dump(data, ini["directory"]["dataset"]+"/waveform_data/%d"%(filenum), compress=3)

            window_shift = int(ini["signal"]["window_shift"])
            window_size = int(ini["signal"]["window_size"])
            fft_point = int(ini["signal"]["fft_point"])
            if ini["signal"]["window"] == "Hann":
                window = signal.hann(window_size)
            # window
            renew_data = np.zeros(len(data))
            for j in range(0, len(data), window_shift):
                if j+window_size < len(data):
                    renew_data[j:j+window_size] += data[j:j+window_size] * np.array(window)
            diff = len(data)%fft_point
            tmp = np.abs(fft(renew_data)/(len(renew_data)/2))
            renew_data = np.concatenate((renew_data, np.zeros(diff)),0)
            spectrogram = []
            for j in range(fft_point, len(renew_data), fft_point):
                tmp = renew_data[j-fft_point:j]
                spectrum = np.abs(fft(tmp) / (len(tmp)/2.0))
                spectrogram.append(spectrum)
            spectrogram = np.array(spectrogram)
            if spectrogram.shape[0] > max_len:
                max_len = spectrogram.shape[0]
            # tmp
            joblib.dump(spectrogram, ini["directory"]["dataset"]+"/spectrogram/%d"%(filenum), compress=3)

        # padding
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        dirs = sorted(dirs)
        spectrums = []
        for directory in dirs:
            spectrum = joblib.load(directory)
            diff = max_len - spectrum.shape[0]
            spectrum = np.concatenate((spectrum, np.zeros((diff, spectrum.shape[1]))), 0)
            spectrums.append(spectrum)
        data = torch.Tensor(np.array(spectrums))
        label = torch.LongTensor(list(range(data.shape[0]))).view(-1, 1)
        dataset = TensorDataset(label, data)
        dataloader = DataLoader(dataset)
        hoge.recreate_dir([ini["directory"]["dataset"]+"/spectrogram"])
        #joblib.dump(dataloader, ini["directory"]["dataset"]+"/spectrogram/dataloader", compress=3)
        f = open(ini["directory"]["dataset"]+"/spectrogram/dataloader", "w")
        pickle.dump(dataloader, f)

