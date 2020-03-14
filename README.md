# tacotron
## prerequire

- mecabのinstall

日本語を扱うttsを想定しているため, mecabを用いて形態素解析

```
brew install mecab
brew install mecab-ipadic
brew install swig
pip install mecab-python3
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


