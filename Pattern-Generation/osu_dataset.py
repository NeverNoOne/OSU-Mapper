import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from numpy import ndarray
from typing import Any
import osu

#TODO self.BeatMaps should be the output fromat --> what output format do I want/need from the Transformer
#TODO testing if it works
class OsuBeatmapDataset(Dataset):
    def __init__(self, map_dir, transform=None, target_transform=None) -> None:
        '''
        parameters
        ---
        map_dir -> the directory which contains the beatmap folders
        '''
        maps = osu.BeatMap.getMaps_from_Dir(map_dir)
        self.audio_labels = self.__generate_labels__(maps)
        self.BeatMaps = maps
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.audio_labels)
    
    def __getitem__(self, idx:int):
        return self.BeatMaps[idx].HitObjects

    def __generate_labels__(self, maps:list[osu.BeatMap]):
        '''
        returns a tuple of [bpm, another tuple[onset_strenghts, onset_times]]
        '''
        values:list[tuple[Any, tuple[ndarray,ndarray]]] = []
        values.clear()
        print("generating audio labels...")
        count = 0
        for map in maps:
            AA = map.General.GetAudioAnalyser()
            bpm, _ = AA.Get_Beattrack()
            onsets = AA.Get_Onsets()
            values.append((bpm, onsets))
            AA = None
            count += 1
            print(f'{100 * count/len(maps):0.2f}%')
        print("audio labels generated")
        return values
    
if __name__ == '__main__':
    import tracemalloc
    tracemalloc.start()
    ds = OsuBeatmapDataset('Maps')
    current, peak = tracemalloc.get_traced_memory()
    print(f"{current / 1024 / 1024:.2f}mb")
    print(f"{peak / 1024 / 1024:.2f}mb")
    print(f'generated {len(ds)} audio labels')
    tracemalloc.stop()