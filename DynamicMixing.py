import random
import numpy as np
import librosa
import os
import soundfile as sf

from scipy import signal


class DynamicMixing:
    def __init__(self,
                 bg_noise_dataset,
                 bb_noise_dataset,
                 rir_dataset,
                 snr_range,
                 sir_range,
                 sr = 16000,
                 max_bg_noise_to_mix = 3,
                 max_speakers_to_mix = 3,
                 reverb_proportion = 0.5,
                 target_level = -25,
                 target_level_floating_value = 10,
                 allowed_overlapped_bg_noise = True,
                 silence_length = 0.2,
                 saved_dir = None
                 ):
        """
        Dynamic mixing
        Args:
            bg_noise_dataset: A text file containing list of background noise audios
            bb_noise_dataset: A text file containing list of bubble noise (speech of a speaker) audios
            rir_dataset: A text file containing list of Room Impulse Response audios
            snr_range: Background noise level. Default is [-5, 25]
            sir_range: Bubble noise level. Default is [-5, 25]
            sr: Sample rate. Default is 16000
            max_bg_noise_to_mix: The maximum number of BACKGROUND noises added to the clean audio when <allowed_overlapped_bg_noise> is True.
                                 If <allowed_overlapped_bg_noise> is False, add UNLIMITED number of BACKGROUND noises till the end of the 
                                 audio with silence between. Default is 3
            max_speakers_to_mix: The maximum number of speakers appear in the clean audio (bubble noise). Default is 3
            reverb_proportion: Chance of using reverb. Default is 0.5
            target_level: Default is -25
            target_level_floating_value: Default is 10
            allowed_overlapped_bg_noise: Whether to allow overlapped BACKGROUND noise. 
                                         If False, evenly add UNLIMITED number of BACKGROUND noises with silence between. Default is True.
                                         * Note: For BUBBLE noise, overlapped is allowed by default. 
                                                 Since if we add a new speaker to the clean speech audio, 
                                                 there exits two overlapped voice in the audio
            silence_length: length of silence between every two consecutive BACKGROUND noise. Only used when <allowed_overlapped_bg_noise> is False. Default is 0.2
            saved_dir: The directory to save the generated noisy audio. Default is None
        """
        super().__init__()
        # acoustic args
        self.sr = sr
        self.max_bg_noise_to_mix = max_bg_noise_to_mix
        self.max_speakers_to_mix = max_speakers_to_mix
        self.reverb_proportion = reverb_proportion
        self.target_level = target_level
        self.target_level_floating_value = target_level_floating_value
        self.allowed_overlapped_bg_noise = allowed_overlapped_bg_noise
        self.silence_length = int(silence_length * self.sr)
        self.saved_dir = saved_dir

        assert 0 <= self.reverb_proportion <= 1, "reverberation proportion should be in [0, 1]"

        noise_provided = False
        if bg_noise_dataset is not None:
            assert os.path.exists(bg_noise_dataset)
            self.bg_noise_dataset_list = [line.rstrip('\n') for line in open(bg_noise_dataset, "r")]
            noise_provided = True
        else:
            self.bg_noise_dataset_list = []

        if bb_noise_dataset is not None:
            assert os.path.exists(bb_noise_dataset)
            self.bb_noise_dataset_list = [line.rstrip('\n') for line in open(bb_noise_dataset, "r")]
            noise_provided = True
        else:
            self.bb_noise_dataset_list = []

        if rir_dataset is not None:
            assert os.path.exists(rir_dataset)
            self.rir_dataset_list = [line.rstrip('\n') for line in open(rir_dataset, "r")]
        else:
            self.rir_dataset_list = []

        assert noise_provided == True, "You must provide either the bg_noise_dataset or bb_noise_dataset argument."

        self.snr_list = self.parse_range(snr_range)
        self.sir_list = self.parse_range(sir_range)

    def parse_range(self, snr_range):
        assert len(snr_range) == 2, f"The range of SNR should be [low, high], not {snr_range}."
        assert snr_range[0] <= snr_range[-1], f"The low SNR should not larger than high SNR."

        low, high = snr_range
        snr_list = []
        for i in range(low, high + 1, 1):
            snr_list.append(i)

        return snr_list

    def random_select_from(self, dataset_list):
        return random.choice(dataset_list)

    def norm_amplitude(self, data, scalar=None, eps=1e-6):
        '''
        Normalize the audio energy
        args:
            - data: audio data from librosa.load(). Ít should be 1D data.
            - scalar: If provided, audio will be normed by this value.
            - eps: Avoid divide by Zero errors.
        '''
        if not scalar:
            scalar = np.max(np.abs(data)) + eps

        return data / scalar, scalar

    def audiowrite(self, destpath, audio, sample_rate=16000, norm=False, target_level=-25, \
                    clipping_threshold=0.99, clip_test=False):
        '''Function to write audio'''

        if clip_test:
            if self.is_clipped(audio, clipping_threshold=clipping_threshold):
                raise ValueError("Clipping detected in audiowrite()! " + \
                                destpath + " file not written to disk.")

        if norm:
            audio = self.normalize(audio, target_level)
            max_amp = max(abs(audio))
            if max_amp >= clipping_threshold:
                audio = audio/max_amp * (clipping_threshold)

        destpath = os.path.abspath(destpath)
        destdir = os.path.dirname(destpath)

        if not os.path.exists(destdir):
            os.makedirs(destdir)

        sf.write(destpath, audio, sample_rate)
        return


    def rescale(self, data, target_level=-25, eps=1e-6):
        '''
        Rescale the audio energy to a target db
        args:
            - data: audio data from librosa.load(). Ít should be 1D data.
            - target_level: target energy that you want to rescale .
            - eps: Avoid divide by Zero errors.
        '''
        rms = np.sqrt(np.mean(data ** 2))
        scalar = 10 ** (target_level / 20) / (rms + eps)
        data *= scalar
        return data, rms, scalar


    def is_clipped(self, data, clipping_threshold=0.999):
        '''
        Check if any audio energy is greater than the threshold
        '''
        return any(np.abs(data) > clipping_threshold)


    def load_wav(self, path, sr=16000):
        return librosa.load(path,  sr=sr)[0]

    def subsample(self, data, sub_sample_length):
        """
        args:
            - data: audio data from librosa.load(). Ít should be 1D data.
            - sub_sample_length: length of audio sub sample. If sub_sample_length > len(data), padding will be used.
        """
        assert np.ndim(data) == 1, f"Only support 1D data. The dim is {np.ndim(data)}"
        length = len(data)

        if length > sub_sample_length:
            start = np.random.randint(length - sub_sample_length)
            end = start + sub_sample_length
            data = data[start:end]
            assert len(data) == sub_sample_length
            return data
        elif length < sub_sample_length:
            data = np.append(data, np.zeros(sub_sample_length - length, dtype=np.float32))
            return data
        else:
            return data

    def select_noise_y(self, target_length, start_pos):
        target_length = int(target_length)
        start_pos = int(start_pos)
        noise_y = np.zeros(target_length, dtype=np.float32)

        noise_file = self.random_select_from(self.bg_noise_dataset_list)
        noise_to_add = self.load_wav(noise_file, sr=self.sr)

        if self.allowed_overlapped_bg_noise:
            if len(noise_to_add) < target_length:
                idx_start = np.random.randint(target_length - len(noise_to_add))
                noise_y[idx_start:idx_start + len(noise_to_add)] += noise_to_add
            else:
                noise_y = noise_to_add[:target_length]
        else:
            if start_pos + len(noise_to_add) < target_length:
                noise_y[start_pos:start_pos + len(noise_to_add)] += noise_to_add

                start_pos += len(noise_to_add)
            else:
                noise_y[start_pos:] += noise_to_add[:target_length-start_pos]

                start_pos = -1
        
            if start_pos != -1:
                if start_pos + self.silence_length < target_length:
                    start_pos = start_pos + self.silence_length
                else:
                    start_pos = -1
        return noise_y, start_pos, noise_file

    def select_speaker_y(self, target_length):
        speaker_file = self.random_select_from(self.bb_noise_dataset_list)
        speaker_y = np.zeros(target_length, dtype=np.float32)
        speaker_to_added = self.load_wav(speaker_file, sr=self.sr)

        if len(speaker_to_added) < target_length:
            idx_start = np.random.randint(target_length - len(speaker_to_added))
            speaker_y[idx_start:idx_start + len(speaker_to_added)] += speaker_to_added
        else:
            speaker_y = speaker_to_added[:target_length]

        return speaker_y, speaker_file

    def mix(self, clean_y, sirs, speakers_y, noises_y, snrs, rir=None, eps=1e-6):
        """
        args:
            - clean_y: Clean audio data
            - sirs: List of corresponding sir values applied to the speakers_y
            - speakers_y: list of bubble noise audios
            - noises_y: list of background noise audios
            - snrs: List of corresponding snr values applied to the noises_y
            - rir: Room Impulse Response audio
            - eps
        """
        if rir is not None:
            if rir.ndim > 1:
                rir_idx = np.random.randint(0, rir.shape[0])
                rir = rir[rir_idx, :]
            clean_y = signal.fftconvolve(clean_y, rir)[:len(clean_y)]
            
        noisy_y = np.zeros(len(clean_y), dtype=np.float32)

        clean_y, _ = self.norm_amplitude(clean_y)
        clean_y, _, _ = self.rescale(clean_y, self.target_level)
        clean_rms = (clean_y ** 2).mean() ** 0.5

        noisy_y += clean_y

        # Mix bubble nosie
        for speaker_y, sir in zip(speakers_y, sirs):
            speaker_y, _ = self.norm_amplitude(speaker_y)
            speaker_y, _, _ = self.rescale(speaker_y, self.target_level)
            speaker_rms = (speaker_y ** 2).mean() ** 0.5

            sir_scalar = clean_rms / (10 ** (sir / 20)) / (speaker_rms + eps)
            speaker_y *= sir_scalar
            noisy_y  += speaker_y

        # Mix background noise
        for noise_y, snr in zip(noises_y, snrs):
            noise_y, _ = self.norm_amplitude(noise_y)
            noise_y, _, _ = self.rescale(noise_y, self.target_level)
            noise_rms = (noise_y ** 2).mean() ** 0.5

            snr_scalar = clean_rms / (10 ** (snr / 20)) / (noise_rms + eps)
            noise_y *= snr_scalar
            noisy_y  += noise_y


        # Randomly select RMS value between -15 dBFS and -35 dBFS and rescale noisy speech with that value
        noisy_target_level = np.random.randint(
            self.target_level - self.target_level_floating_value,
            self.target_level + self.target_level_floating_value
        )
        noisy_y, _, noisy_scalar = self.rescale(noisy_y, noisy_target_level)
        clean_y *= noisy_scalar


        # check if there are any amplitudes exceeding +/- 1. If so, normalize all the signals accordingly
        if self.is_clipped(noisy_y):
            noisy_y_scalar = np.max(np.abs(noisy_y)) / (0.99 - eps)  # 相当于除以 1
            noisy_y = noisy_y / noisy_y_scalar
            clean_y = clean_y / noisy_y_scalar

        return noisy_y, clean_y

    def generate(self, clean_path, save_to_dir = False):
        clean_filename = clean_path.split('/')[-1]
        clean_y = self.load_wav(clean_path)        

        sirs = []
        speakers_y = []
        bb_noise_files = []
        if len(self.bb_noise_dataset_list) > 0:
            n_speakers = np.random.randint(1, self.max_speakers_to_mix+1)
            while len(speakers_y) < n_speakers:
                speaker_y, bb_noise_file = self.select_speaker_y(len(clean_y))
                speakers_y += [speaker_y]
                sirs += [self.random_select_from(self.sir_list)]
                bb_noise_files += [bb_noise_file]

        snrs = []
        noises_y = []
        bg_noise_files = []
        if len(self.bg_noise_dataset_list) > 0:
            if self.allowed_overlapped_bg_noise:
                n_noises = np.random.randint(1, self.max_bg_noise_to_mix+1)
                while len(noises_y) < n_noises:
                    noise_y, _, bg_noise_file = self.select_noise_y(len(clean_y), -1)
                    noises_y += [noise_y]
                    snrs += [self.random_select_from(self.snr_list)]
                    bg_noise_files += [bg_noise_file]
            else:
                start_pos = 0
                while start_pos != -1:
                    noise_y, start_pos, bg_noise_file = self.select_noise_y(len(clean_y), start_pos)
                    noises_y += [noise_y]
                    snrs += [self.random_select_from(self.snr_list)]
                    bg_noise_files += [bg_noise_file]

        '''
        3 cases:
            - Mix background noise only
            - Mix bubble noise only
            - Mix both
        '''
        if len(self.bg_noise_dataset_list) > 0 and len(self.bb_noise_dataset_list) > 0:
            choice = np.random.randint(3)
            if choice == 1:
                noises_y = []
                snrs = []
            if choice == 2:
                speakers_y = []
                sirs = []

        use_reverb = bool(np.random.random(1) < self.reverb_proportion)
        if use_reverb and len(self.rir_dataset_list) > 0:
            rir_file = self.random_select_from(self.rir_dataset_list)
            rir = self.load_wav(rir_file, sr=self.sr)
        else:
            rir_file, rir = None, None

        noisy_y, clean_y = self.mix(
            clean_y = clean_y,
            speakers_y = speakers_y,
            noises_y = noises_y,
            snrs = snrs,
            sirs = sirs,
            rir = rir
        )

        noisy_y = noisy_y.astype(np.float32)
        clean_y = clean_y.astype(np.float32)

        if save_to_dir:
            saved_path = os.path.join(self.saved_dir, clean_filename)
            self.audiowrite(saved_path, noisy_y)


        output = {
            "noisy": noisy_y,
            "bg_noise_files": bg_noise_files,
            "bb_noise_files": bb_noise_files,
            "sirs": sirs,
            "snrs": snrs,
            "rir_file": rir_file
        }
        print(output)
        return output