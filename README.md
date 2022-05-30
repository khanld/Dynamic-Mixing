# DYNAMIC MIXING (MIX-ON-THE-FLY)

### Documentation
An easy-to-use Dynamic mixing code for Speech Processing task such as: Speech Enhancement, Speech Source Separation, Target Speech Extraction, Speech Augmentation,...
</br>
All documents related to this repo can be found here:
- [DNS-Challenge](https://huggingface.co/docs/transformers/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC)

### Installation
```
pip install -r requirements.txt
```

### Usage
It is recommended to understand the DynamicMixing arguments before using it
<br>
Inline python code:
```python
from DynamicMixing import DynamicMixing

mixer = DynamicMixing(bg_noise_dataset = 'DM/audios/bg_noise.txt',
                      bb_noise_dataset = 'DM/audios/bb_noise.txt',
                      rir_dataset = 'DM/audios/rir.txt',
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
                      saved_dir = 'DM/audios/noisy')

clean_path = 'DM/audios/clean book_00000_chp_0009_reader_06709_2.wav'
output = mixer.generate(clean_path, save_to_dir = True)

# output is a dictionary, pls check the DynamicMixing code
print(output)

# get the noisy
noisy_y = output['noisy']
print(noisy_y)
```

Generate and save noisy audios:
```CMD
python generate.py \
    --clean_dataset DM/audios/clean.txt
    --bg_noise_dataset DM/audios/bb_noise.txt
    --bb_noise_dataset DM/audios/bg_noise.txt
    --rir_dataset DM/audios/rir.txt
    --snr_range [-5,25]
    --sir_range [-5,25]
    --max_bg_noise_to_mix 3
    --max_speakers_to_mix 3
    --reverb_proportion 0.5
    --target_level -25
    --target_level_floating_value 10
    --allowed_overlapped_bg_noise True
    --silence_length 0.2
    --saved_dir DM/audios/noisy
```


