import numpy as np
import multiprocessing as mp

from numba import njit
from scipy.io import wavfile
from fastdtw import fastdtw

import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch

import tqdm
from . import istarmap
from .speech_utils import load_wav_file, normalize_signal, normalize_spectogram, \
     compute_melspectogram, find_speech_in_signal, find_speech_melspectogram, dBFS

def plot_path(gt, pred, path, start_a=0, start_b=0, **plt_kwargs):
    
    fig = plt.figure(**plt_kwargs)

    ax1 = plt.subplot(211)
    ax2 = plt.subplot(212)

    # create connection line between subpl not
    
    for p in path[::25]:
        con = ConnectionPatch(xyA=(p[0]+start_a, 0), xyB=(p[1]+start_b, 0), coordsA="data", coordsB="data",
                        axesA=ax1, axesB=ax2, color="red")
        ax1.add_artist(con)

    ax1.set_ylabel("ground truth")
    ax1.imshow(gt[::-1,:], cmap='inferno', extent=[0, gt.shape[1], 0, gt.shape[0]])
    ax2.set_ylabel("prediction")
    ax2.imshow(pred[::-1,:], cmap='inferno', extent=[0, pred.shape[1], 0, pred.shape[0]])
    plt.show()

@njit
def _f_dist(x, y):
    """
    Calculate distance between frame x and frame y
    Arguments:
        x: Numpy array with shape (Time, Channels)
        y: Numpy array with shape (Time, Channels)
    Return 
        distance
    """
    return np.sqrt(np.mean(np.square(x - y)))

def _eval_fn(gt, pred, config):
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
    
    sample_rate = config.get('sampling_rate', 22050)
    n_fft = config.get('num_fft', 1024)
    window_size = config.get('window_size', 1024)
    hop_length = config.get('hop_length', 256)
    energy_threshold = config.get('energy_threshold', -450)
    n_mels = config.get('num_features', 80)
    
    if isinstance(gt, str) and isinstance(pred, str):
        gt_signal = load_wav_file(gt, sample_rate)
        pred_signal = load_wav_file(pred, sample_rate)
    elif isinstance(gt, np.ndarray) and isinstance(pred,  np.ndarray):
        gt_signal = gt
        pred_signal = pred
    else:
        raise ValueError("input must be string (path) or numpy array")

    if np.min(gt_signal) < -1 or np.max(gt_signal) > 1:
        gt_signal = normalize_signal(gt_signal)
        
    if np.min(pred_signal) < -1 or np.max(pred_signal) > 1:
        pred_signal = normalize_signal(pred_signal)
    
    
    gt_speech = find_speech_in_signal(gt_signal, sample_rate, window_size, hop_length)
    pred_speech = find_speech_in_signal(pred_signal, sample_rate, window_size, hop_length)
    
    # ### Gain same energy ###
    gt_dbfs = np.mean(dBFS(gt_speech, window_size, hop_length))
    pred_dbfs = np.mean(dBFS(pred_speech, window_size, hop_length))
    gain = gt_dbfs - pred_dbfs
    pred_signal = np.clip(pred_signal*(10**(gain/20)),
                            a_min=-1.0, a_max=1.0)
    
    gt_spec = compute_melspectogram(gt_signal, sample_rate, n_fft, window_size, hop_length, n_mels=n_mels)
    pred_spec = compute_melspectogram(pred_signal, sample_rate, n_fft, window_size, hop_length, n_mels=n_mels)
    
    gt_start_frame, gt_end_frame = find_speech_melspectogram(gt_spec, sample_rate, n_fft, energy_threshold=energy_threshold)
    pred_start_frame, pred_end_frame = find_speech_melspectogram(pred_spec, sample_rate, n_fft, energy_threshold=energy_threshold)
    
    gt_spec = normalize_spectogram(gt_spec[:, gt_start_frame:gt_end_frame])
    pred_spec = normalize_spectogram(pred_spec[:, pred_start_frame:pred_end_frame])
    
    distance, path = fastdtw(gt_spec.T, pred_spec.T, dist=_f_dist)
    return distance/len(path), path, (gt_start_frame, gt_end_frame), (pred_start_frame, pred_end_frame)
    
class EvalDTW():
    
    """ 
    ** Work on CPU only **
    
    num_processes : number of threads
    sample_rate : sample rate of audio data
    log_energy_threshold : energy threshold for detect speech energy
    
    """
    
    def __init__(self, config):
        
        self.num_processes = config['eval_config'].get("num_processes", 6)
        self.audio_config = config['audio_config']
        
        if self.num_processes is None:
            self.pool = None
        else:
            self.pool = mp.Pool(self.num_processes)
    
    def eval_fn(self, gt, pred):
        return _eval_fn(gt, pred, self.audio_config)
    
    def eval_parallel(self, gt_paths, pred_paths):
        """
        Calculate DTW evaluate in parallel

        Arguments:
            gt_paths: List of wavfile path
            pred_paths: List of wavfile path

        Return:
            distances : List of distance between predictions and groundtruths
            paths : List of optimal warpping paths
            position of speech in groundtruth : List of tuples
            position of speech of prediction : List of tuples
        """
        assert self.num_processes is not None, "Parameter 'num_processes' is not assigned"
        assert len(gt_paths) == len(pred_paths), "Size must equal!"
        
        dists = list()
        paths = list()
        gt_speech_pos = list()
        pred_speech_pos = list()
        
        iterable = [(gt_paths[i], pred_paths[i], self.audio_config ) for i in range(len(gt_paths))]
        for result in tqdm.tqdm(self.pool.istarmap(_eval_fn, iterable), total=len(iterable)):
            dists.append(result[0])
            paths.append(result[1])
            gt_speech_pos.append(result[2])
            pred_speech_pos.append(result[3])
        
        return dists, paths, gt_speech_pos, pred_speech_pos
        
if __name__ == "__main__":
    from scipy.io import wavfile
    import time
    
    gt_wav_path = [
        "/Users/thananchai/Projects/ASR/newera/ThaiTTS/tests/ref_wav/01 มีพระบรมราชโองการโปรดเกล้าโปรดกระหม่อมพระราชทานยศทหาร.wav",
        "/Users/thananchai/Projects/ASR/newera/ThaiTTS/tests/ref_wav/03 ศึกแมนเชสเตอร์ดาร์บี้ในฟุตบอลพรีเมียร์ลีกอังกฤษ.wav",
        "/Users/thananchai/Projects/ASR/newera/ThaiTTS/tests/ref_wav/13 ไก่งามเพราะขน คนจนเพราะกรรมขี่จรวด.wav"
    ]
    
    pred_wav_path = [
        "/Users/thananchai/Projects/ASR/newera/ThaiTTS/outputs/26tests/210420/wav/temp_90000_0.wav",
        "/Users/thananchai/Projects/ASR/newera/ThaiTTS/outputs/26tests/210420/wav/temp_102000_2.wav",
        "/Users/thananchai/Projects/ASR/newera/ThaiTTS/outputs/26tests/210420/wav/temp_102000_12.wav"
    ]
    
    configs = {
    "audio_config": {
        "sampling_rate": 22050,
        "num_features": 80,
        "num_fft": 1024,
        "window_size": 1024,
        "hop_length": 256,
        "ste_threshold": 1.0,
        "zcr_threshold": 0.1,
        "energy_threshold": -450,
        "energy_band": [100, 1000],
        "filter_length": 1024
    },

    "eval_config": {
        "num_processes":6
    }
}
    
    tts_eval = EvalDTW(configs)
    
    ###########################
    start = time.time()
    dist, path, start_gt_idx, start_pred_idx = tts_eval.eval_fn(gt_wav_path[0], pred_wav_path[0])
    print("Distance : {} ".format(dist))
    print("Time used : {} ".format(time.time()-start))
    
    ###########################
    print("TEST Audio Inputs")
    start = time.time()
    dist, path, start_gt_idx, start_pred_idx = tts_eval.eval_fn(wavfile.read(gt_wav_path[0])[1], 
                                                                wavfile.read(pred_wav_path[0])[1])
    
    print("Distance : {} ".format(dist))
    print("Time used : {} ".format(time.time()-start))
    
    ###########################
    print("TEST PARALLEL EVALUTION")
    
    start = time.time()
    
    dist = tts_eval.eval_parallel(gt_wav_path, pred_wav_path)[0]
    print("Average Distance : {} ".format(np.mean(dist)))
    print("Time used : {} ".format(time.time()-start))
    ###########################