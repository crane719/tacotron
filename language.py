import MeCab
import numpy as np

class Ja():
    def __init__(self):
        self.hira_list = ['あ','い','う','え','お','か','き','く','け','こ','さ','し','す','せ','そ','た','ち','つ','て','と','な','に','ぬ','ね','の','は','ひ','ふ','へ','ほ','ま','み','む','め','も','や','ゆ','よ','ら','り','る','れ','ろ','わ','を','ん','っ','ゃ','ゅ','ょ','ー','が','ぎ','ぐ','げ','ご','ざ','じ','ず','ぜ','ぞ','だ','ぢ','づ','で','ど','ば','び','ぶ','べ','ぼ','ぱ','ぴ','ぷ','ぺ','ぽ']
        self.hira_list = self.hira_list + ["、", "。"]
        self.kata_list = ['ア','イ','ウ','エ','オ','カ','キ','ク','ケ','コ','サ','シ','ス','セ','ソ','タ','チ','ツ','テ','ト','ナ','ニ','ヌ','ネ','ノ','ハ','ヒ','フ','ヘ','ホ','マ','ミ','ム','メ','モ','ヤ','ユ','ヨ','ラ','リ','ル','レ','ロ','ワ','ヲ','ン','ッ','ャ','ュ','ョ','ー','ガ','ギ','グ','ゲ','ゴ','ザ','ジ','ズ','ゼ','ゾ','ダ','ヂ','ヅ','デ','ド','バ','ビ','ブ','ベ','ボ','パ','ピ','プ','ペ','ポ']
        self.kata_list = self.kata_list + ["、", "。"]

    def mecab_list(self,text):
        tagger = MeCab.Tagger("-Ochasen")
        tagger.parse('')
        node = tagger.parseToNode(text)
        word_class = []
        while node:
            word = node.surface
            wclass = node.feature.split(',')
            if wclass[0] != u'BOS/EOS':
                if wclass[6] == None:
                    word_class.append((word,wclass[0],wclass[1],wclass[2],"", wclass[7]))
                else:
                    word_class.append((word,wclass[0],wclass[1],wclass[2],wclass[6], wclass[7]))
            node = node.next
        return word_class

    def kana2label(self, text):
        text = self.kana2kata(text)
        labels = []

        for char in list(text):
            arg = list(np.where(np.array(self.kata_list)==char))[0]
            if arg or arg==[0]:
                labels.append(arg[0])
        return labels


    def kana2hira(self,text):
        text = self.kana2kata(text)
        renew_text = ""
        for char in list(text):
            arg = list(np.where(np.array(self.kata_list)==char))[0]
            if arg or arg==[0]:
                renew_text += self.hira_list[arg[0]]
            else:
                renew_text += char
        return renew_text

    def kana2kata(self,text):
        text_class = self.mecab_list(text)
        renew_text = ""
        for word_class in text_class:
            word = word_class[5]
            renew_text += word
        return renew_text

    @property
    def get_hiradict():
        return self.hira_list
