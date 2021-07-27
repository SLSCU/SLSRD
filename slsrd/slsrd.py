import numpy as np
import multiprocessing as mp

from fastdtw import fastdtw

import librosa

import tqdm
from .asr_model import ASRModel
from .istarmap import istarmap
from .speech_utils import normalize_signal, normalize, compute_spectrogram, compute_melspectogram, gain_energy

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
            self._asr_model = None
            
        if self.use_spectral == False and self._asr_model is None:
            raise ValueError("if not use ASR feature, use_spectral must be True")
    
    def eval_fn(self, gt, pred, gt_asr_feature_path=None, pred_asr_feature_path=None):

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

        pred_signal = gain_energy(gt_signal, pred_signal, self.sample_rate, 
                                  self.window_size, self.hop_size, self.db_threshold)
        
        # Extract spectrogram
        if self.use_spectral == True:
            gt_feature = compute_spectrogram(gt_signal, self.sample_rate, n_fft=self.n_fft, 
                                            window_size=self.window_size, hop_size=self.hop_size)
            gt_feature = normalize(gt_feature[:self.num_features, :]).T
            
            pred_feature = compute_spectrogram(pred_signal, self.sample_rate, n_fft=self.n_fft, 
                                            window_size=self.window_size, hop_size=self.hop_size)
            pred_feature = normalize(pred_feature[:self.num_features, :]).T
        else:
            gt_feature, pred_feature = (None, None)
        
        # Extract ASR feature
        if gt_asr_feature_path is not None and pred_asr_feature_path is not None:
            gt_asr_ft = np.load(gt_asr_feature_path)
            pred_asr_ft = np.load(pred_asr_feature_path)
        elif self._asr_model is not None:
            gt_asr_mel = compute_melspectogram(gt_signal, self.sample_rate, window_size=self.window_size,
                                                hop_size=self.hop_size, n_fft=self.n_fft, num_features=64)
            gt_asr_mel = normalize(gt_asr_mel).T

            pred_asr_mel = compute_melspectogram(pred_signal, self.sample_rate, window_size=self.window_size,
                                                    hop_size=self.hop_size, n_fft=self.n_fft, num_features=64)
            pred_asr_mel = normalize(pred_asr_mel).T
                
            gt_asr_ft = self._asr_model.get_asr_feature(gt_asr_mel)
            pred_asr_ft = self._asr_model.get_asr_feature(pred_asr_mel)
        else:
            gt_asr_ft, pred_asr_ft = (None, None)
        
        
        if gt_asr_ft is None and gt_feature is not None: # SD
            dist, path = fastdtw(gt_feature, pred_feature, dist=_dist)
            return dist/len(path)
        elif gt_asr_ft is not None and gt_feature is None: # LSRD
            gt_asr_ft = normalize(gt_asr_ft[0].astype(np.float32))
            pred_asr_ft = normalize(pred_asr_ft[0].astype(np.float32))
            
            dist, path = fastdtw(gt_asr_ft, pred_asr_ft, dist=_dist)
            return dist/len(path)
        elif gt_asr_ft is not None and gt_feature is not None: # SLSRD
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
            return dist/len(path)
        

class TTSEval():
    def __init__(self, config):
        self.num_worker = config.get("num_worker", 10)
        self.eval_method = SLSRD(config)
        
        if config['asr_params']['checkpoint'] is None:
            self.pool = mp.Pool(self.num_worker)
        else:
            self.pool = mp.pool.ThreadPool(self.num_worker)

    def eval(self, gt_paths, pred_paths):

        """
            Calculate DTW evaluate in parallel

            Arguments:
                gt_paths: List of wavfile path
                pred_paths: List of wavfile path

            Return:
                distances : List of distance between predictions and groundtruths
        """
        assert self.num_worker is not None, "Parameter 'num_worker' is not assigned"
        assert len(gt_paths) == len(pred_paths), "Size must equal!"
        
        dists = list()
        iterable = [(gt_paths[i], pred_paths[i] ) for i in range(len(gt_paths))]
        for result in tqdm.tqdm(self.pool.istarmap(self.eval_method.eval_fn, iterable), total=len(iterable)):
            dists.append(result)

        return dists