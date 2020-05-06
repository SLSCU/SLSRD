
import argparse
import numpy as np
import json
import pandas as pd
import multiprocessing as mp
import librosa

from scipy.io import wavfile
from tts_eval import EvalDTW
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='TTS Evaluation')
    parser.add_argument("csv", type=str, help='csv file')
    parser.add_argument("--output", type=str, default='result.csv', help='output')
    parser.add_argument('--config_file', type=str, default='config.json',
                        help='configuration file')
    args = parser.parse_args()
    
    config_f = args.config_file
    input_csv = args.csv
    output_csv = args.output
    
    config = json.load(open(config_f))
    
    tts_eval = EvalDTW(config)
    
    df = pd.read_csv(input_csv)
    
    print(f"Input File : {input_csv}")
    print(f"Output File : {output_csv}")
    print(f"Total : {len(df)}")
    
    D, _, _, _ = tts_eval.eval_parallel(df['groundtruth_wav'], df['synthesis_wav'])
    
    df['distance'] = D
    
    df.to_csv(output_csv, index=False)  
    