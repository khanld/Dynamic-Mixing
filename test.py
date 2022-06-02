from DynamicMixing import DynamicMixing

mixer = DynamicMixing(bg_noise_dataset = 'audios/bg_noise.txt',
                      bb_noise_dataset = 'audios/bb_noise.txt',
                      rir_dataset = 'audios/rir.txt',
                      snr_range = [-5, 25],
                      sir_range = [-5, 25],
                      sr = 16000,
                      max_bg_noise_to_mix = 3,
                      max_speakers_to_mix = 3,
                      reverb_proportion = 0.5,
                      target_level = -25,
                      target_level_floating_value = 10,
                      allowed_overlapped_bg_noise = True,
                      silence_length = 0.2,
                      saved_dir = 'audios/noisy')

clean_path = 'audios/clean/book_00000_chp_0009_reader_06709_2.wav'
output = mixer.generate(clean_path, save_to_dir = True)

# output is a dictionary, pls check the DynamicMixing code
print("Output: ", output)

# get the noisy
noisy_y = output['noisy']
print("Noisy data: ", noisy_y)