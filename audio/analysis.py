import librosa
import numpy as np
from typing import Any
import soundfile as sf
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

Analyser = AudioAnalyser('Maps/785731 S3RL - Catchit (Radio Edit)/audio.mp3')

tempo, beat_frames = Analyser.Get_Beattrack()
beat_times = librosa.frames_to_time(beat_frames, sr=Analyser.sr)
onset_env = librosa.onset.onset_strength(y=Analyser.y, sr=Analyser.sr)

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
