import numpy as np
import multiprocessing as mp

from fastdtw import fastdtw

import librosa

import tqdm
from .asr_model import ASRModel
from .istarmap import istarmap
from .speech_utils import  normalize_signal, compute_spectrogram, compute_melspectogram, dBFS

import torch
from torch.nn.functional import interpolate

def _dist(x, y):
    """
    Calculate distance between frame x and frame y
    Arguments:
        x: Numpy array with shape (Time, Channels)
        y: Numpy array with shape (Time, Channels)
    Return 
        distance
    """
    return np.sqrt(np.mean(np.square(x - y)))

def normalize(x):
    return (x-x.mean())/(x.std()+1e-10)

class SLSRD():
    
    def __init__(self, config):
        
        self.sample_rate = config.get('sampling_rate', 22050)
        self.n_fft = config.get('num_fft', None)
        self.window_size = config.get('window_size', 0.02)
        self.hop_size = config.get('hop_length', 0.01)
        self.db_threshold = config.get('db_threshold', 50)
        self.num_features = config.get('num_features', 64)
        self.use_spectral = config.get('use_spectral', True)

        asr_checkpoint = config['asr_params'].get('checkpoint', None)
        if asr_checkpoint is not None:
            self._asr_model = ASRModel(asr_checkpoint, 
                                    config['asr_params'].get('device', 'cpu'))
        else:
            raise ValueError("ASR checkpoint is not found")
            
    def gain_ref_energy(self, ref_signal, signal):
        ### Gain same energy ###
        rel_non_silent_interval = librosa.effects.split(ref_signal, top_db=self.db_threshold, 
                                frame_length=int(self.sample_rate * self.window_size), 
                                hop_length=int(self.sample_rate * self.hop_size))
        signal_non_silent_interval = librosa.effects.split(signal, top_db=self.db_threshold, 
                                frame_length=int(self.sample_rate * self.window_size), 
                                hop_length=int(self.sample_rate * self.hop_size))

        ref_speech = np.concatenate([ref_signal[non_silent_interval[0]:non_silent_interval[1]] 
                                    for non_silent_interval in rel_non_silent_interval])
        speech = np.concatenate([signal[non_silent_interval[0]:non_silent_interval[1]] 
                                    for non_silent_interval in signal_non_silent_interval])

        gt_dbfs = np.mean(dBFS(ref_speech, int(self.sample_rate * self.window_size), 
                            int(self.sample_rate * self.hop_size)))
        pred_dbfs = np.mean(dBFS(speech, int(self.sample_rate * self.window_size), 
                            int(self.sample_rate * self.hop_size)))
        gain = gt_dbfs - pred_dbfs
        signal = np.clip(signal*(10**(gain/20)),
                        a_min=-1.0, a_max=1.0)
        return signal
    
    def eval_fn(self, gt, pred):

        """
        eval_fn is used for evaluate a pair of prediction and groundtruth
        
        Arguments:
            gt: Path of wavfile or Numpy array of signal with shape (Time)
            pred: Path of wavfile or Numpy array of signal with shape (Time)

        Return:
            distance : Normalize distance between prediction and groundtruth
            path : Optimal warpping path 
            position of speech in groundtruth : A tuple (start_index, end_index)
            position of speech of prediction : A tuple (start_index, end_index)
        """

        if isinstance(gt, str) and isinstance(pred, str):
            gt_signal, _ = librosa.load(gt, self.sample_rate)
            pred_signal, _ = librosa.load(pred, self.sample_rate)
            
        elif isinstance(gt, np.ndarray) and isinstance(pred,  np.ndarray):
            gt_signal = gt
            pred_signal = pred
            gt_signal = normalize_signal(gt_signal)
            pred_signal = normalize_signal(pred_signal)
        else:
            raise ValueError("input must be string (path) or numpy array")

        gt_signal, _ = librosa.effects.trim(gt_signal, top_db=self.db_threshold, 
                                frame_length=int(self.sample_rate * self.window_size), 
                                hop_length=int(self.sample_rate * self.hop_size))

        pred_signal, _ = librosa.effects.trim(pred_signal, top_db=self.db_threshold, 
                                frame_length=int(self.sample_rate * self.window_size), 
                                hop_length=int(self.sample_rate * self.hop_size))

        pred_signal = self.gain_ref_energy(gt_signal, pred_signal)
        

        gt_asr_mel = compute_melspectogram(gt_signal, self.sample_rate, window_size=self.window_size,
                                                hop_size=self.hop_size, n_fft=self.n_fft, num_features=64)
        gt_asr_mel = normalize(gt_asr_mel).T

        pred_asr_mel = compute_melspectogram(pred_signal, self.sample_rate, window_size=self.window_size,
                                                hop_size=self.hop_size, n_fft=self.n_fft, num_features=64)
        pred_asr_mel = normalize(pred_asr_mel).T
            

        gt_asr_ft = self._asr_model.get_asr_feature(gt_asr_mel)
        pred_asr_ft = self._asr_model.get_asr_feature(pred_asr_mel)


        dist = 0

        if self.use_spectral:
            gt_feature = compute_spectrogram(gt_signal, self.sample_rate, n_fft=self.n_fft, 
                                            window_size=self.window_size, hop_size=self.hop_size)
            gt_feature = normalize(gt_feature[:self.num_features, :]).T
            
            pred_feature = compute_spectrogram(pred_signal, self.sample_rate, n_fft=self.n_fft, 
                                            window_size=self.window_size, hop_size=self.hop_size)
            pred_feature = normalize(pred_feature[:self.num_features, :]).T

            gt_asr_ft = interpolate(
                torch.from_numpy(gt_asr_ft.astype(np.float32)).transpose(1,2), 
                size=[len(gt_feature)], mode='linear').transpose(1,2).numpy()[0]

            pred_asr_ft = interpolate(
                torch.from_numpy(pred_asr_ft.astype(np.float32)).transpose(1,2), 
                size=[len(pred_feature)], mode='linear').transpose(1,2).numpy()[0]

            gt_asr_ft = normalize(gt_asr_ft.astype(np.float32))
            pred_asr_ft = normalize(pred_asr_ft.astype(np.float32))

            dist, path = fastdtw(np.concatenate([gt_feature, gt_asr_ft], 1),
                            np.concatenate([pred_feature, pred_asr_ft], 1), dist=_dist)
        else:
            gt_asr_ft = normalize(gt_asr_ft[0].astype(np.float32))
            pred_asr_ft = normalize(pred_asr_ft[0].astype(np.float32))


            dist, path = fastdtw(gt_asr_ft, pred_asr_ft, dist=_dist)

        return dist/len(path)

class TTSEval():
    def __init__(self, config):
        self.num_thread = config.get("num_thread", 6)
        self.eval_method = SLSRD(config)

        self.pool = mp.pool.ThreadPool(self.num_thread)

    def eval(self, gt_paths, pred_paths):

        """
            Calculate DTW evaluate in parallel

            Arguments:
                gt_paths: List of wavfile path
                pred_paths: List of wavfile path

            Return:
                distances : List of distance between predictions and groundtruths
        """
        assert self.num_thread is not None, "Parameter 'num_thread' is not assigned"
        assert len(gt_paths) == len(pred_paths), "Size must equal!"
        
        dists = list()
        iterable = [(gt_paths[i], pred_paths[i] ) for i in range(len(gt_paths))]
        for result in tqdm.tqdm(self.pool.istarmap(self.eval_method.eval_fn, iterable), total=len(iterable)):
            dists.append(result)

        return dists