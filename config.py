# hyper parameter
encoder_filter_set = list(range(1, 17))
decoder_filter_set = list(range(1, 9))

# fft parameter
# sample点数を指定する必要あり
# fftのサンプリングなどから計算
import scipy
window = scipy.signal.hann(N)
p = 0.97 # pre emphasis

