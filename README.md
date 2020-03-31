# tacotron
## prerequire

- mecabのinstall

日本語を扱うttsを想定しているため, mecabを用いて形態素解析. macの場合は以下

```bash:
brew install mecab
brew install mecab-ipadic
brew install swig
pip install mecab-python3
```

- pythonのpackageのinstall

```bash:
pip install -r requrements.txt
```

- datasetのinstall

https://www.kaggle.com/bryanpark/japanese-single-speaker-speech-dataset/data を使って学習を行っている.  
directory内にdatasetをダウンロード


## train

以下のコマンドで学習. 前処理を行いデータセットを作成する場合は引数としてpreprocessを指定. 学習した重みを読み込み直し、学習を行う場合は、loadを指定

```bash:
python train.py --preprocess --load
```

## architecture

### CBHG

1dconv, highway, GRUから成る

strideは1

全てのconv層でbatchnorm

1. 1~Kサイズのfilterでconv(Kgram的な役割を有する)

2. 出力をstackし, maxpool1d
3. 固定長のconv
4. highway net
5. bidirectional GRU

### encoder

テキストの連続的な表現を抽出する目的

入力はcharacter sequence(onehotやembedded)

embedするためにprenetと呼ばれる非線型変換を行う

prenetからの出力を, attentionを用いたCBHGにより表現ベクトルに変換

### decoder
content-based tanh attention decoder
コンテキストベクトルとattention RNNの出力をconcat

decoderではresidual

80bandメルスケールのスペクトログラムをターゲットとして利用

後処理ネットワークを用いてseq2seqターゲットより波形に変換

各デコーダのステップで, 複数(rフレーム)の重複しない出力フレームを予測

最初のステップでは全てが0のフレームを想定(GO frame)

step tの予測したrを最後のフレームを次のデコーダの入力とする

### loss
ロスはL1. seq2seqデコーダ及び後処理netの出力でロスを算出


