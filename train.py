import preprocess as pp
import models
import configparser

is_preprocess = False

ini = configparser.ConfigParser()
ini.read("./config.ini")

if is_preprocess:
    pp.Preprocess()

tacotron = models.Tacotron()

for i in range(1, int(ini["hyperparameter"]["epoch_num"])+1):
    tacotron.train(i)
