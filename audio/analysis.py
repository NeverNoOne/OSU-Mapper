import librosa
import numpy as np
from typing import Any
import soundfile as sf
import matplotlib.pyplot as plt

class AudioAnalyser:
    def __init__(self, song_path):
        self.song_path = song_path
        self.y, self.sr = librosa.load(song_path)
        pass

    def Get_Beattrack(self, onset_env:bool = False) -> tuple[Any, np.ndarray]:
        '''
        returns tempo in bpm, beat frames
        '''
        if onset_env:
            onset = librosa.onset.onset_strength(y=self.y, sr=self.sr)
            return librosa.beat.beat_track(onset_envelope=onset, sr=self.sr)
        else:
            return librosa.beat.beat_track(y=self.y, sr=self.sr)
        
    def Get_Beats_from_onset(self) -> tuple[np.ndarray, np.ndarray]:
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        times = librosa.times_like(onset_env, sr=self.sr)
        mean = np.mean(onset_env)
        beat_array = np.copy(onset_env)
        for index in range(onset_env.size):
            if onset_env[index] < mean:
                beat_array[index] = 0
        return times, beat_array
    
    def Get_Pitch(self):
        f0, voiced_flag, voiced_prob = librosa.pyin(y=self.y, fmin=float(librosa.note_to_hz('C2')), fmax=float(librosa.note_to_hz('C7')))
        times = librosa.times_like(f0, sr=self.sr)
        return times, f0

    def Show_onset_beats(self):
        tempo, beat_frames = self.Get_Beattrack(False)
        beat_times = librosa.frames_to_time(beat_frames, sr=self.sr)
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        times = librosa.times_like(onset_env, sr=self.sr)
        #compute mean
        mean = np.mean(onset_env)
        beat_array = np.copy(onset_env)
        for index in range(onset_env.size):
            if onset_env[index] < mean:
                beat_array[index] = 0

        plt.plot(times, onset_env, label='Onset Strength')
        #plt.vlines(beat_times, 0, np.max(onset_env), color='r', alpha=0.75, linestyle='--', label='Beats')
        plt.plot(times, beat_array, label='Beats', color='r')
        plt.legend(loc='upper right')
        plt.xlabel('Time (s)')
        plt.title('Onset Strength and Beat Locations')
        plt.show()

    def Show_fourier(self):
        tempo_features = librosa.feature.fourier_tempogram(y=self.y, sr=self.sr)
        librosa.display.specshow(tempo_features, sr=self.sr, x_axis='time', y_axis='tempo', cmap='cool')
        plt.title('Fourier Tempogram')
        plt.colorbar()
        plt.show()

Analyser = AudioAnalyser('Maps/785731 S3RL - Catchit (Radio Edit)/audio.mp3')

# times, f0 = Analyser.Get_Pitch()
# plt.figure(figsize=(10,6))
# plt.plot(times, f0, label='F0', color='b')
# plt.xlabel('Time (s)')
# plt.ylabel('Frequency (Hz)')
# plt.title('Pitch Tracking with pYIN')
# plt.legend()
# plt.show()

#Analyser.Show_onset_beats()
#Analyser.Show_fourier()

# #adding beats whenever onset > average
# hit_sound, hit_sr = librosa.load('Maps/785731 S3RL - Catchit (Radio Edit)/soft-hitclap.wav')

# hit_duration = len(hit_sound) / hit_sr

# onset_env = librosa.onset.onset_strength(y=Analyser.y, sr=Analyser.sr)
# onset_avg = np.mean(onset_env)

# frames_above_avg = np.where(onset_env > onset_avg)[0]
# times_above_avg = librosa.frames_to_time(frames_above_avg, sr=Analyser.sr)

# sample_indices = (times_above_avg * Analyser.sr).astype(int)

# output_audio = Analyser.y.copy()

# for idx in sample_indices:
#     end_idx = idx + len(hit_sound)
#     if end_idx <= len(output_audio):
#         output_audio[idx:end_idx] += hit_sound
# output_audio = np.clip(output_audio, -1.0, 1.0)

# sf.write('test.wav', output_audio, Analyser.sr)

# _, beats = Analyser.Get_Beattrack(False)

# hit_sound, ht_sr = librosa.load('Maps/785731 S3RL - Catchit (Radio Edit)/soft-hitclap.wav')

# y_with_hits = np.copy(Analyser.y)

# for beat in librosa.frames_to_samples(beats):
#     if beat + len(hit_sound) < len(y_with_hits):
#         y_with_hits[beat:beat+len(hit_sound)] += hit_sound

# y_with_hits = y_with_hits / np.max(np.abs(y_with_hits))

# sf.write("test.wav", y_with_hits, Analyser.sr)
# import matplotlib.pyplot as plt
# fig, ax = plt.subplots(nrows=3, sharex=True)

# librosa.display.waveshow(y=Analyser.y, sr=Analyser.sr, ax=ax[0])
# ax[0].set(title='mono')
# ax[0].label_outer()

# librosa.display.waveshow(y=y_with_hits, sr=Analyser.sr, ax=ax[1])
# ax[1].set(title='mono')
# ax[1].label_outer()

# plt.show()
