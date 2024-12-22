import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Any
import osu
from audio.analysis import AudioAnalyser

#TODO adding difficulty to labels and modify __getitem__ for it
class OsuBeatmapDataset(Dataset):
    def __init__(self, map_dir, transform=None, target_transform=None, count_mapsets=None) -> None:
        """
        Parameters:
        ---
        map_dir : str
            The directory containing beatmap folders.
        transform : callable, optional
            Optional transform to apply to input features.
        target_transform : callable, optional
            Optional transform to apply to target labels.
        """
        maps = [x for x in osu.BeatMap.getMaps_from_Dir(map_dir, count_mapsets=count_mapsets) if x.General.mode == osu.Mode_Type.osu]
        self.audio_labels = self.__generate_labels__(maps)
        self.note_placements = [[int(y.time) for y in x.HitObjects] for x in maps]
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx:int):
        _, onsets = self.audio_labels[idx]
        note_placement = self.note_placements[idx]

        if self.transform:
            onsets = self.transform(onsets)
        if self.target_transform:
            note_placement = self.target_transform(note_placement)

        return torch.tensor(onsets), torch.tensor(note_placement)

    def __generate_labels__(self, maps:list[osu.BeatMap]):
        '''
        returns a tuple of [bpm, ndarray[onset_strenghts, onset_times]]
        '''
        values:list[tuple[Any, np.ndarray]] = []
        values.clear()
        print("generating audio labels...")
        count = 0
        AA:AudioAnalyser
        oldAudioFileName:str = ''
        oldTitle:str = ''
        for map in maps:
            if map.General.AudioFilename != oldAudioFileName or map.Metadata.Title != oldTitle:
                oldAudioFileName = map.General.AudioFilename
                oldTitle = map.Metadata.Title
                AA = map.General.GetAudioAnalyser()
                bpm, _ = AA.Get_Beattrack()
                onset_strenghts, onset_times = AA.Get_Onsets()
            values.append((bpm, np.vstack((onset_strenghts, onset_times)))) # type: ignore
            #del AA
            count += 1
            print(f'{100 * count/len(maps):0.2f}%')
            print ("\033[A\033[A")
        print("audio labels generated")
        return values
    
if __name__ == '__main__':
    import tracemalloc
    tracemalloc.start()
    ds = OsuBeatmapDataset('Maps')
    dl = DataLoader(ds)
    current, peak = tracemalloc.get_traced_memory()
    print(f"{current / 1024 / 1024:.2f}mb")
    print(f"{peak / 1024 / 1024:.2f}mb")
    print(f'generated {len(ds)} audio labels')
    tracemalloc.stop()
    # lbm = osu.BeatMap.getMaps_from_MapDir('Maps/839864 S3RL - Catchit (Radio Edit)')
    # bm = lbm[0]
    # AA = bm.General.GetAudioAnalyser()
    # onset_strenghts, onset_times = AA.Get_Onsets()
    # print(onset_strenghts.shape)
    # print(onset_times.shape)
    # combined = np.vstack((onset_strenghts, onset_times))
    # print(combined.shape)