# Spectral and Latent Speech Representation Distortion for TTS Evaluation
## How to Use

Example
```
python run_eval.py "input csv file format" \
                   --ref_path_col  "column name of reference path in csv" \
                   --syn_path_col "column name of systhesized path in csv" \
                   --output "output csv file format"  \
                   --config "config file"
```

If you want to extract ASR feature and store it on disk, you can use the script ```extract_feature.py```

Example
```
python extract_feature.py "input csv file format" \
                          --ref_path_col  "column name of reference path in csv" \
                          --syn_path_col "column name of systhesized path in csv" \
                          --feat_path "Directory to save ASR feature" \
                          --csv_output "output csv file format"  \
                          --config "config file"
```

then use the output csv from ```extract_feature.py``` as input csv for ```run_eval.py```, and add argument ```--ref_asr_feature_path_col``` and ```--syn_asr_feature_path_col```

Example
```
python run_eval.py "input csv file format" \
                   --ref_path_col  "column name of reference path in csv" \
                   --syn_path_col "column name of systhesized path in csv" \
                   --syn_asr_feature_path_col "column name of asr feature reference path in csv" \
                   --syn_asr_feature_path_col "column name of asr feature systhesized path in csv" \
                   --output "output csv file format"  \
                   --config "config file"
```

## Example parameter in config file

```

{
    "sampling_rate": 16000, /* sampling rate of audio*/
    "num_features": 200, /* number of spectrogram features */ 
    "window_size": 0.02,
    "hop_length": 0.01,
    "db_threshold":35, /* DB threshold for remove silence */
    "asr_params": {
        "checkpoint":"", /* path of wav2letter+ checkpoint */
        "device":"cpu"
    },
    "num_worker":30
}

```

Metadata of wav2letter+ pretrain without hovorod has provide in folder 'w2lplus_eng_meta'
