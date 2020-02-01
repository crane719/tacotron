import preprocess as pp
import models
import configparser

is_preprocess = False

ini = configparser.ConfigParser()
ini.read("./config.ini")

if is_preprocess:
    pp.Preprocess()

tacorron = models.Tacotron()

