import models
import configparser
import hoge
import shutil

is_load = True

ini = configparser.ConfigParser()
ini.read("./config.ini")

if hoge.is_dir_existed("eval_result"):
    print("delete file...")
    print("- eval_result")
    shutil.rmtree("./eval_result")

required_dirs = ["eval_result"]
hoge.make_dir(required_dirs)

tacotron = models.Tacotron(is_load)
tacotron.evaluate(mode="eval")
