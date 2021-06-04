# Spectral and Latent Speech Representation Distortion for TTS Evaluation
## HOW TO USE

```
python run_eval.py "input csv file format" --ref_path_col  "column name of reference path in csv" --syn_path_col "column name of systhesized path in csv" --output "output csv file format"  --config "config file"
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
    "num_thread":30
}

```

Metadata of wav2letter+ pretrain without hovorod has provide in folder 'w2lplus_eng_meta'
