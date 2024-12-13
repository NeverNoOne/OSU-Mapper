import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import osu

class OsuBeatmapDataset(Dataset):
    def __init__(self, map_dir, transform=None, target_transform=None) -> None:
        '''
        parameters
        ---
        map_dir -> the directory which contains the beatmap folders
        '''
        osu.BeatMap.getMaps_from_Dir(map_dir)
        self.transform = transform
        self.target_transform = target_transform