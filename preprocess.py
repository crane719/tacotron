import soundfile as sf
import numpy as np
import hoge
import configparser
import joblib

class Preprocess():
    def __init__(self):
        print("Start preprocess:")
        # read config
        ini = configparser.ConfigParser()
        ini.read("./config.ini")

        # read raw data
        print("     read raw data...")
        filedirs = hoge.get_filedirs(ini["directory"]["raw_data"]+"*")
        rates = []
        datas = []
        max_len = 0
        for filedir in filedirs:
            data, rate = sf.read(filedir)
            rates.append(rate)
            datas.append(data)

            if len(data) > max_len:
                max_len = len(data)

        print("     pad raw data...")
        datas = [ np.concatenate([data, np.zeros(max_len - len(data))]) for data in datas]
        datas = np.array(datas)

        print("     dump data...")
        joblib.dump(datas, ini["directory"]["dataset"]+"data", compress=3)


