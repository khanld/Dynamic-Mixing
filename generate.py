import librosa
import os
import argparse

from distutils.util import strtobool
from DynamicMixing import DynamicMixing

if __name__ == '__main__':
    args = argparse.ArgumentParser(description='GENERATE MIXTURE')
    args.add_argument('--clean_dataset', type=str, required = True,
                      help='a text file containing list of clean speech audio paths')
    args.add_argument('--bg_noise_dataset', type=str, default = '/home/khanhld/Desktop/DynamicMixing/audios/bg_noise.txt',
                      help='Default is None')
    args.add_argument('--bb_noise_dataset', type=str, default = '/home/khanhld/Desktop/DynamicMixing/audios/bb_noise.txt',
                      help='Default is None')
    args.add_argument('--rir_dataset', type=str, default = '/home/khanhld/Desktop/DynamicMixing/audios/rir.txt',
                      help='Default is None')
    args.add_argument('--snr_range', type=lambda x: [int(item) for item in x.split(',')], default = "-5,25",
                      help='Background noise level. Default is [-5, 25].')
    args.add_argument('--sir_range', type=lambda x: [int(item) for item in x.split(',')], default = "-5,25",
                      help='Bubble noise level. Default is [-5, 25]')                  
    args.add_argument('--max_bg_noise_to_mix', type=int, default = 3,
                      help='Default is 3')   
    args.add_argument('--max_speakers_to_mix', type=int, default = 3,
                      help='Default is 3')   
    args.add_argument('--reverb_proportion', type=float, default = 0.5,
                      help='Default is 0.5')   
    args.add_argument('--target_level', type=int, default = -25,
                      help='Default is -25')   
    args.add_argument('--target_level_floating_value', type=int, default = 10,
                      help='Default is 10')  
    args.add_argument('--allowed_overlapped_bg_noise', type=lambda x: bool(strtobool(x)), default = "true", 
                      help = 'Default is true')    
    args.add_argument('--silence_length', type=float, default = 0.2, 
                      help = 'Default is 0.2')    
    args.add_argument('--saved_dir', type=str, required = True)   

    args = args.parse_args()
    clean_ds = args.clean_dataset
    del(args.clean_dataset)

    print(vars(args))
    
    mixer = DynamicMixing(**vars(args))
    for clean_path in [line.rstrip('\n') for line in open(clean_ds, "r")]:
        mixer.generate(clean_path, save_to_dir=True)
    

