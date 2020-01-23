import preprocess as pp
#import models
import configparser

is_preprocess = True

ini = configparser.ConfigParser()
ini.read("./config.ini")

if is_preprocess:
    pp.Preprocess()

#tacorron = models.Tacotron()
