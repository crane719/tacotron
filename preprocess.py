import soundfile as sf
import numpy as np
import hoge
import configparser
import joblib
import language
from scipy import signal

class Preprocess():
    def __init__(self):
        print("Start preprocess:")
        # read config
        ini = configparser.ConfigParser()
        ini.read("./config.ini")

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

        print("     dump data...")
        if hoge.is_dir_existed(ini["directory"]["dataset"]):
            hoge.recreate_dir([ini["directory"]["dataset"]])
        else:
            hoge.make_dir([ini["directory"]["dataset"]])
        joblib.dump([datas, labels], ini["directory"]["dataset"]+"/data", compress=3)


