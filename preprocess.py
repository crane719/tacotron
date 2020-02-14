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
        max_len = 0
        data_len = len(raw_dataconf)
        while len(raw_dataconf):
            if i%100==0:
                print("     [%d/%d]"%(i, data_len))
            col = raw_dataconf.pop()

            if len(col) == 0:
                break
            text = col.split("|")[1]
            label = ja.kana2label(text)

            if len(label) > max_len:
                max_len = len(label)
            joblib.dump(torch.LongTensor(label), ini["directory"]["dataset"]+"/label_data/%d"%(i), compress=3)
            i += 1

        # padding
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/label_data/*")
        for i,directory in enumerate(dirs):
            label = joblib.load(directory)
            diff = max_len - label.shape[0]
            label = np.concatenate((label, np.ones((diff))*len(ja.get_hiralist)), 0)
            joblib.dump(torch.LongTensor(label), ini["directory"]["dataset"]+"/label_data/%d"%(i), compress=3)

        print("     signal processing")
        filedirs = hoge.get_filedirs(ini["directory"]["raw_data"]+"/*")
        max_len = 0
        max_value = 0
        max_mel_len = 0
        max_mel_value = 0
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
            fs = int(ini["signal"]["samplingrate"])
            if ini["signal"]["window"] == "Hann": window = signal.hann(window_size)
            # window
            renew_data = np.zeros(len(data))
            for j in range(0, len(data), window_shift):
                if j+window_size < len(data):
                    renew_data[j:j+window_size] += data[j:j+window_size] * np.array(window)
            diff = len(data)%fft_point
            tmp = np.abs(fft(renew_data)/(len(renew_data)/2))
            renew_data = np.concatenate((renew_data, np.zeros(diff)),0)

            # no voice cut
            no_voice_args = list(np.where(np.abs(renew_data)<np.max(np.abs(renew_data))/1000))[0]
            renew_data = np.delete(renew_data, no_voice_args)
            """
            spectrogram = []
            for j in range(fft_point, len(renew_data), fft_point):
                tmp = renew_data[j-fft_point:j]
                spectrum = np.abs(fft(tmp) / (len(tmp)/2.0))
                spectrogram.append(spectrum)
            spectrogram = np.array(spectrogram)
            """
            f, t, spectrogram = scipy.signal.spectrogram(renew_data, fs=fs, nfft=fft_point)
            spectrogram = spectrogram.T
            joblib.dump(spectrogram, ini["directory"]["dataset"]+"/spectrogram/%d"%(filenum), compress=3)
            if spectrogram.shape[0] > max_len:
                max_len = spectrogram.shape[0]
            tmp_max_value = np.max(spectrogram.reshape(-1))
            if tmp_max_value > max_value:
                max_value = tmp_max_value
            """
            import matplotlib.pyplot as plt
            f, t, spectrogram = scipy.signal.spectrogram(renew_data, fs=fs, nfft=fft_point)
            plt.pcolormesh(spectrogram*100000)
            plt.savefig("spectrogram.png")
            plt.close()
            joblib.dump(spectrogram, ini["directory"]["dataset"]+"/spectrogram/%d"%(filenum), compress=3)
            """

            # melspectrogram
            mel_spectrogram = librosa.feature.melspectrogram(renew_data,
                                                             S=spectrogram.T,
                                                             sr=int(ini["signal"]["samplingrate"]),
                                                             n_mels=int(ini["signal"]["band"]))
            mel_spectrogram = mel_spectrogram.T
            joblib.dump(mel_spectrogram, ini["directory"]["dataset"]+"/melspectrogram/%d"%(filenum), compress=3)
            if mel_spectrogram.shape[0] > max_mel_len:
                max_mel_len = mel_spectrogram.shape[0]
            if max(mel_spectrogram.reshape(-1)) > max_mel_value:
                max_mel_value = max(mel_spectrogram.reshape(-1))

        # spectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        for directory in dirs:
            spectrogram = joblib.load(directory)
            diff = max_len - spectrogram.shape[0]
            spectrogram = np.concatenate((spectrogram, np.zeros((diff, spectrogram.shape[1]))), 0)
            spectrogram /= max_value
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        # melspectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/melspectrogram/*")
        for directory in dirs:
            spectrogram = joblib.load(directory)
            diff = max_mel_len - spectrogram.shape[0]
            spectrogram = np.concatenate((spectrogram, np.zeros((diff, spectrogram.shape[1]))), 0)
            spectrogram /= max_mel_value
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

