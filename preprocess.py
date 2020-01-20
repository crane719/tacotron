import soundfile as sf
import numpy as np
import hoge
import configparser
import joblib
import language
from scipy import signal
from scipy.fftpack import fft, fftshift
import math

class Preprocess():
    def __init__(self):
        # read config
        ini = configparser.ConfigParser()
        ini.read("./config.ini")

        # create required directory
        if hoge.is_dir_existed(ini["directory"]["dataset"]):
            hoge.recreate_dir([ini["directory"]["dataset"]])
        else:
            hoge.make_dir([ini["directory"]["dataset"]])
        print("\n")

        print("Start preprocess:")

        # read raw data
        print("     read raw data...")
        filedirs = hoge.get_filedirs(ini["directory"]["raw_data"]+"/*")
        datas = []
        labels = []
        max_len = 0
        for filedir in filedirs:
            data, rate = sf.read(filedir)

            # resampling
            wanted_rate_bias = int(ini["signal"]["samplingrate"])/rate
            wanted_data_len = int(len(data) * wanted_rate_bias)
            data = signal.resample(data, wanted_data_len)
            datas.append(data)
            if len(data) > max_len:
                max_len = len(data)

        # signal discription
        ja = language.Ja()
        f = open(ini["directory"]["raw_dataconf"], "r")
        raw_dataconf = f.read()
        for col in raw_dataconf.split("\n"):
            text = col.split("|")[1]
            label = ja.kana2label(text)
            labels.append(label)

        print("     pad raw data...")
        datas = [ np.concatenate([data, np.zeros(max_len - len(data))]) for data in datas]
        datas = np.array(datas)

        print("     dump waveform data...")
        joblib.dump(datas, ini["directory"]["dataset"]+"/waveform_data", compress=3)
        joblib.dump(labels, ini["directory"]["dataset"]+"/label_data", compress=3)

        print("     fft...")
        window_shift = int(ini["signal"]["window_shift"])
        window_size = int(ini["signal"]["window_size"])
        fft_point = int(ini["signal"]["fft_point"])
        if ini["signal"]["window"] == "Hann":
            window = signal.hann(window_size)
        # window
        spectrograms = []
        for i, data in enumerate(datas):
            renew_data = np.zeros(len(data))
            for j in range(0, len(data), window_shift):
                if j+window_size < len(data):
                    renew_data[j:j+window_size] += data[j:j+window_size] * np.array(window)
            datas[i] = renew_data
            diff = len(data)%fft_point
            tmp = np.abs(fft(renew_data)/(len(renew_data)/2))
            renew_data = np.concatenate((renew_data, np.zeros(diff)),0)
            for j in range(fft_point, len(renew_data), fft_point):
                tmp = renew_data[j-fft_point:j]
                spectrum = np.abs(fft(tmp) / (len(tmp)/2.0))
                if j == fft_point:
                    spectrogram = spectrum.reshape((1, -1))
                else:
                    spectrogram = np.concatenate((spectrogram, spectrum.reshape((1, -1))), 0)
            spectrograms.append(spectrogram)
            #sf.write("./%d.wav"%(i), renew_data, int(ini["signal"]["samplingrate"]))
        joblib.dump(spectrograms, ini["directory"]["dataset"]+"/spectrogram", compress=3)


