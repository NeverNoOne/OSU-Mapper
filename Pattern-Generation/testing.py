import torch
from osu_dataset import OsuBeatmapDataset

ds = OsuBeatmapDataset('Maps')
x, y = ds[0]
print(f'x: {x.shape} | {x}')
print(f'y: {y.shape} | {y}')
# for x in ds:
#     x, y = x
#     print(f'[]input | output: {x.shape} | {y.shape}')