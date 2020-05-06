# HOW TO USE

python run_eval.py input_csv --output output_csv --config config_file

input_csv : csv file has 'groundtruth_wav' presenting paths of groundtruth speech wav file and 'synthesis_wav' presenting paths of synthesis speech wav file.
output_csv : csv file has 'groundtruth_wav' presenting paths of groundtruth speech wav file, 'synthesis_wav' presenting paths of synthesis speech wav file and 'distance' presenting distance between groundtruth and synthesis. Default is 'result.csv'.
config_file : config file has parameters use for evaluation and processing. Default is 'config.json'.

if you want to run it faster, you can increase the number in 'num_processes' parameter for more compute processes in config file.
