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

        ave = []
        mel_ave = []
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

            """
            # window
            renew_data = np.zeros(len(data))
            for j in range(0, len(data), window_shift):
                if j+window_size < len(data):
                    renew_data[j:j+window_size] += data[j:j+window_size] * np.array(window)
            diff = len(data)%fft_point
            tmp = np.abs(fft(renew_data)/(len(renew_data)/2))
            renew_data = np.concatenate((renew_data, np.zeros(diff)),0)
            """

            # no voice cut
            #no_voice_args = list(np.where(np.abs(renew_data)<np.max(np.abs(renew_data))/20))[0]
            #renew_data = np.delete(renew_data, no_voice_args)
            #renew_data = np.delete(data, no_voice_args)
            """
            spectrogram = []
            for j in range(fft_point, len(renew_data), fft_point):
                tmp = renew_data[j-fft_point:j]
                spectrum = np.abs(fft(tmp) / (len(tmp)/2.0))
                spectrogram.append(spectrum)
            spectrogram = np.array(spectrogram)
            """

            #f, t, spectrogram = scipy.signal.spectrogram(renew_data, fs=fs, nperseg=fft_point, window=("hann"), mode="magnitude")
            spectrogram = librosa.core.stft(renew_data, n_fft=fft_point, hop_length=window_shift, win_length=window_size, window='hann')
            spectrogram = librosa.amplitude_to_db(np.abs(spectrogram))
            spectrogram = spectrogram.T
            del_args = np.where(np.max(spectrogram, axis=1) < th)[0]
            spectrogram = np.delete(spectrogram, del_args, axis=0)

            s = librosa.db_to_amplitude(spectrogram).T
            #spectrogram = librosa.util.normalize(spectrogram)
            #spectrogram = (spectrogram-np.average(spectrogram))/np.var(spectrogram)
            joblib.dump(spectrogram, ini["directory"]["dataset"]+"/spectrogram/%d"%(filenum), compress=3)

            if spectrogram.shape[0] > max_len:
                max_len = spectrogram.shape[0]
            #tmp_max_value = np.max(spectrogram.reshape(-1))
            #if tmp_max_value > max_value:
            #    max_value = tmp_max_value
            ave.append(np.average(spectrogram))
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
                                                             S=s,
                                                             sr=rate,#int(ini["signal"]["samplingrate"]),
                                                             n_mels=int(ini["signal"]["band"]))
            mel_spectrogram = librosa.amplitude_to_db(np.abs(mel_spectrogram))
            mel_spectrogram = mel_spectrogram.T
            #mel_spectrogram = librosa.util.normalize(mel_spectrogram)
            #del_args = np.where(np.max(mel_spectrogram, axis=1) < th)[0]
            #mel_spectrogram = np.delete(mel_spectrogram, del_args, axis=0)
            #mel_spectrogram = (mel_spectrogram-np.average(mel_spectrogram))/np.var(mel_spectrogram)
            joblib.dump(mel_spectrogram, ini["directory"]["dataset"]+"/melspectrogram/%d"%(filenum), compress=3)
            if mel_spectrogram.shape[0] > max_mel_len:
                max_mel_len = mel_spectrogram.shape[0]
            #if max(mel_spectrogram.reshape(-1)) > max_mel_value:
            #    max_mel_value = max(mel_spectrogram.reshape(-1))
            mel_ave.append(np.average(mel_spectrogram))

        # spectrogram padding and normalization
        """
        ave = np.average(ave)
        mel_ave = np.average(mel_ave)
        var = []
        mel_var = []
        print("     calc var...")
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        for directory in dirs:
            spectrogram = joblib.load(directory)
            #spectrogram /= max_value
            #spectrogram -= ave
            var.append(np.average((spectrogram-ave)**2))
            #joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        # melspectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/melspectrogram/*")
        for directory in dirs:
            spectrogram = joblib.load(directory)
            #spectrogram /= max_mel_value
            #spectrogram -= mel_ave
            mel_var.append(np.average((spectrogram-mel_ave)**2))
            #joblib.dump(spectrogram, directory, compress=3)

        print("     normalization and no voice cut...")
        var = np.average(var)
        mel_var = np.average(mel_var)
        max_len = 0
        max_mel_len = 0
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        for directory in dirs:
            spectrogram = joblib.load(directory)
            spectrogram = (spectrogram - ave)/var

            #del_args = np.where(np.max(spectrogram, axis=1) < 0.5)[0]
            #spectrogram = np.delete(spectrogram, del_args, axis=0)

            #diff = max_len - spectrogram.shape[0]
            #spectrogram = np.concatenate((spectrogram, np.zeros((diff, spectrogram.shape[1]))), 0)
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)
            if spectrogram.shape[0] > max_len:
                max_len = spectrogram.shape[0]

        # melspectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/melspectrogram/*")
        for directory in dirs:
            spectrogram = joblib.load(directory)
            #diff = max_mel_len - spectrogram.shape[0]
            spectrogram = (spectrogram - mel_ave)/mel_var
            #spectrogram = np.concatenate((spectrogram, np.zeros((diff, spectrogram.shape[1]))), 0)
            #del_args = np.where(np.max(spectrogram, axis=1) < 0.5)[0]
            #spectrogram = np.delete(spectrogram, del_args, axis=0)
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)
            if spectrogram.shape[0] > max_mel_len:
                max_mel_len = spectrogram.shape[0]
        """

        print("     padding...")
        var = []
        mel_var = []
        ave = np.average(ave)
        mel_ave = np.average(mel_ave)
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/spectrogram/*")
        for filenum, directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            spectrogram = joblib.load(directory)
            var.append(np.average((spectrogram-ave)**2))
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        # melspectrogram padding and normalization
        dirs = hoge.get_filedirs(ini["directory"]["dataset"]+"/melspectrogram/*")
        for filenum, directory in enumerate(dirs):
            if (filenum+1)%1000==0:
                print("     [%d/%d]"%(filenum+1, len(dirs)))
            spectrogram = joblib.load(directory)
            mel_var.append(np.average((spectrogram-mel_ave)**2))
            joblib.dump(torch.Tensor(spectrogram), directory, compress=3)

        print("     normalization...")
        #var = np.sqrt(np.average(var))
        #mel_var = np.sqrt(np.average(mel_var))
        var = np.average(var)
        mel_var = np.average(mel_var)
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
