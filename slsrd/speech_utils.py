import math
import librosa
import numpy as np

def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def normalize_signal(signal):
    """
    Normalize float32 signal to [-1, 1] range
    """
    # return signal / (np.max(np.abs(signal)) + 1e-5)
    if max(signal) > 1 or min(signal) < -1:
        return signal / (32768 + 1e-10)
    return signal
    
def dBFS(signal, window_size, hop_length):
    """Compute Decibels relative to full scale ."""
    rms = librosa.feature.rms(signal, frame_length=window_size, hop_length=hop_length)[0]
    return 20*np.log10(rms+1e-6)

def compute_spectrogram(signal, sample_rate, window_size, hop_size, window=np.hanning, n_fft=None):
    if n_fft is None:
        n_fft = 2**math.ceil(math.log2(sample_rate * window_size))

    S = librosa.stft(signal, n_fft=n_fft, hop_length=int(sample_rate * hop_size),
                    win_length=int(sample_rate * window_size), window=window, center=True, pad_mode='constant')
    S = np.square(np.abs(S))
    return S

def compute_melspectogram(signal, sample_rate, window_size, hop_size, num_features=80, n_fft=None):
    if n_fft is None:
        n_fft = 2**math.ceil(math.log2(sample_rate * window_size))

    prem_signal = preemphasis(signal)
    S = librosa.stft(prem_signal, n_fft=n_fft, hop_length=int(sample_rate * hop_size),
                    win_length=int(sample_rate * window_size), window=np.hanning, center=True, pad_mode='constant')
    powerS = np.square(np.abs(S))

    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=num_features,
                    fmin=0.0, fmax=sample_rate/2, htk=False)
    mel_spectogram = np.dot(mel_basis, powerS)
    mel_spectogram = np.log(mel_spectogram + 1e-20)

    return mel_spectogram



