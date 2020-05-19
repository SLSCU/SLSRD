import argparse

import json
import numpy as np
import pandas as pd

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
    
    D, paths, gt_speech_pos, pred_speech_pos = tts_eval.eval_parallel(df['groundtruth_wav'], df['synthesis_wav'])
    
    df['distance'] = D
    df['path_lenght'] = [len(p) for p in paths]
    df['gt_len'] = [gt_pos[1] - gt_pos[0] for gt_pos in gt_speech_pos]
    df['pred_len'] = [pred_pos[1] - pred_pos[0] for pred_pos in pred_speech_pos]
    
    df.to_csv(output_csv, index=False)
    