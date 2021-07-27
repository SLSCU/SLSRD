import argparse
import os
import glob
import json

import pandas as pd
import numpy as np
import multiprocessing as mp

import tqdm

import librosa
from slsrd.asr_model import ASRModel
from slsrd.speech_utils import normalize_signal, normalize, compute_melspectogram, gain_energy

def save_feat(d):
    path, feats = d
    np.save(path, feats)

def main(args):
    config_f = args.config_file
    input_csv = args.csv
    
    config = json.load(open(config_f))
    num_fft = config['window_size'] if config['num_fft'] is None else config['num_fft']
    num_fft = int(config['sampling_rate']*num_fft)

    asr_model = ASRModel(config['asr_params'].get('checkpoint', None), 
                         config['asr_params'].get('device', 'cpu'))
    
    if not os.path.exists(args.feat_path):
        os.makedirs(args.feat_path)

    df = pd.read_csv(input_csv)
    df[args.ref_asr_feature_path_col] = df[args.ref_path_col].apply(lambda x: os.path.join(args.feat_path,
                                                                    os.path.splitext(x)[0].split('/')[-1]+".npy"))
    df[args.syn_asr_feature_path_col] = df[args.syn_path_col].apply(lambda x: os.path.join(args.feat_path,
                                                                    os.path.splitext(x)[0].split('/')[-1]+".npy"))
    
    print('Extracting feature')
    
    feat_d = list()
    for _, row in tqdm.tqdm(df.iterrows(), total=len(df)):
        gt = row[args.ref_path_col]
        pred = row[args.syn_path_col]
        
        gt_signal, _ = librosa.load(gt, config['sampling_rate'])
        pred_signal, _ = librosa.load(pred, config['sampling_rate'])
        
        
        pred_signal = gain_energy(gt_signal, pred_signal, config['sampling_rate'],
                                  config['window_size'], config['hop_length'], config['db_threshold'])
        
        gt_asr_mel = compute_melspectogram(gt_signal, config['sampling_rate'], window_size=config['window_size'],
                                                hop_size=config['hop_length'], n_fft=num_fft, num_features=64)
        gt_asr_mel = normalize(gt_asr_mel).T

        pred_asr_mel = compute_melspectogram(pred_signal, config['sampling_rate'], window_size=config['window_size'],
                                                hop_size=config['hop_length'], n_fft=num_fft, num_features=64)
        pred_asr_mel = normalize(pred_asr_mel).T
            
        gt_asr_ft = asr_model.get_asr_feature(gt_asr_mel)
        pred_asr_ft = asr_model.get_asr_feature(pred_asr_mel)
        
        feat_d.append((row[args.syn_asr_feature_path_col], pred_asr_ft))
        feat_d.append((row[args.ref_asr_feature_path_col], gt_asr_ft))
        
    print('Saving feature')
    pool = mp.Pool(config["num_thread"])
    for _ in tqdm.tqdm(pool.imap(save_feat, feat_d), total=len(feat_d)):
        pass
    
    df.to_csv(args.csv_output)
    print('Done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Script for extracting ASR Feature')
    parser.add_argument("csv", type=str, help='csv file')
    parser.add_argument("--ref_path_col", type=str, help='column name of reference aduio path')
    parser.add_argument("--syn_path_col", type=str, help='column name of synthesis aduio path')
    parser.add_argument("--ref_asr_feature_path_col", type=str, default='ref_asr_feature_path', help='column name of reference asr feature path')
    parser.add_argument("--syn_asr_feature_path_col", type=str, default='syn_asr_feature_path', help='column name of synthesis asr feature path')
    parser.add_argument("--feat_path", type=str, default='feats', help='path for saving feature')
    parser.add_argument("--csv_output", type=str, default='csv_output.csv', help='output csv')
    parser.add_argument('--config_file', type=str, default='config.json', help='configuration file')
    args = parser.parse_args()

    main(args)
                        
                        