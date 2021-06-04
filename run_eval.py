import argparse

import json
import pandas as pd

from slsd import TTSEval
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='TTS Evaluation')
    parser.add_argument("csv", type=str, help='csv file')
    parser.add_argument("--ref_path_col", type=str, help='column name of reference aduio path')
    parser.add_argument("--syn_path_col", type=str, help='column name of synthesis aduio path')
    parser.add_argument("--output", type=str, default='result.csv', help='output')
    parser.add_argument('--config_file', type=str, default='config.json',
                        help='configuration file')
    args = parser.parse_args()
    
    config_f = args.config_file
    input_csv = args.csv
    output_csv = args.output
    
    config = json.load(open(config_f))
    
    tts_eval = TTSEval(config)
    
    df = pd.read_csv(input_csv)
    
    print(f"Input File : {input_csv}")
    print(f"Output File : {output_csv}")
    print(f"Total : {len(df)}")
    
    D = tts_eval.eval(df[args.ref_path_col], 
                      df[args.syn_path_col])
    
    df['distance'] = D
    
    df.to_csv(output_csv, index=False)
    