import numpy as np
from numba import njit

import librosa
from scipy.io import wavfile

def load_wav_file(path, sample_rate=22050):
    sr, signal = wavfile.read(path)
    if sr != sample_rate:
        signal = librosa.resample(signal.astype(np.float32), sr, sample_rate)
    return signal

def normalize_signal(signal):
    return signal/32678.0
    
def normalize_spectogram(S):
    return (S-S.mean())/S.std()
    
def dBFS(signal, window_size, hop_length):
    """Compute Decibels relative to full scale ."""
    rms = librosa.feature.rms(signal, frame_length=window_size, hop_length=hop_length)[0]
    return 20*np.log10(rms)

def zeros_crossing_rate(signal, window_size, hop_length):
    """Compute short-time zero crossing rate."""
    frames = librosa.util.frame(np.append(signal, np.zeros((window_size)), 0), 
                                frame_length=window_size, hop_length=hop_length, axis=0)
    zr = librosa.zero_crossings(frames)
    return np.sum(zr,1)/window_size

def find_speech_in_signal(signal, sample_rate, window_size, hop_length, dbfs_threshold=-50, zcr_threshold=0.2):
    """
    Find speech in signal with dBFS and zeros crossing rate
    """
    dbfs = dBFS(signal, window_size, hop_length)
    zcr = zeros_crossing_rate(signal, window_size, hop_length)
    min_threshold = int(round(0.1*(sample_rate/hop_length))) # 0.1s
    non_silence_indices = np.where((dbfs > dbfs_threshold) & (zcr < zcr_threshold))[0]
    prev_sel_idx = non_silence_indices[0]
    prev_idx = prev_sel_idx
    result_signal = list()
    for cur_idx in non_silence_indices[1:]:
        if abs(prev_idx - cur_idx) > min_threshold or cur_idx == non_silence_indices[-1]: # if silence longer than 0.1s
            sample_idx_1 = prev_sel_idx*hop_length
            sample_idx_2 = cur_idx*hop_length
            if sample_idx_2 > len(signal):
                sample_idx_2 = len(signal)-1
            slice_signal = signal[sample_idx_1:sample_idx_2]
            result_signal+=slice_signal.tolist()
            prev_sel_idx = cur_idx
        prev_idx = cur_idx
            
    return np.array(result_signal)


@njit
def find_speech_melspectogram(S, sample_rate, n_fft, energy_band=(100, 1000), energy_threshold=-450):
    """
    Find start and end position of speech in spectral using energy threshold
    Arguments:
        spec: Numpy array with shape (Channels, Time)
    Return:
        min_idx: Start of speech position
        max_idx: End of speech position
    """
    lower = int(energy_band[0] / (sample_rate / float(n_fft)))
    higher = int(energy_band[1] / (sample_rate / float(n_fft)))

    start_frame = 0
    end_frame = S.shape[1]-1
    
    bandpass = S[lower:higher, :]
    sum_bandpass = bandpass.sum(axis=0)
    above_threshold = np.where(energy_threshold < sum_bandpass)[0]
    if len(above_threshold) > 0:
        start_frame = np.min(above_threshold)
        end_frame = np.max(above_threshold)
    return start_frame, end_frame

def compute_melspectogram(signal, sample_rate, n_fft, window_size, hop_length, n_mels=80):
    S = librosa.stft(y=signal, n_fft=n_fft, hop_length=hop_length, win_length=window_size, 
                         window=np.hanning, pad_mode='constant')
    S = np.square(np.abs(S))   
    S = np.clip(S, a_min=1e-5, a_max=None)
    mel_basis = librosa.filters.mel(sample_rate, n_fft, n_mels=n_mels,
							   fmin=0.0, fmax=8000.0)
    mel_spectogram = np.dot(mel_basis, S)
    mel_spectogram = np.log(mel_spectogram)
    return mel_spectogram