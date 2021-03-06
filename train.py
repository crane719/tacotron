import preprocess as pp
import models
import configparser
import hoge
import shutil

args = hoge.get_args()
is_preprocess = args.preprocess
is_load = args.load

if is_preprocess is None:
    is_preprocess = False
if is_load is None:
    is_load = False

ini = configparser.ConfigParser()
ini.read("./config.ini")

if hoge.is_dir_existed("train_result"):
    print("delete file...")
    print("- train_result")
    shutil.rmtree("./train_result")

required_dirs = ["param", "train_result"]
hoge.make_dir(required_dirs)

if is_preprocess:
    pp.Preprocess()

tacotron = models.Tacotron(is_load)

for epoch in range(1, int(ini["hyperparameter"]["epoch_num"])+1):
    tacotron.train(epoch)
    tacotron.output(epoch)

    if (epoch-1)%20== 0:
        tacotron.evaluate(epoch)

