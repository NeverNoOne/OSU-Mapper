import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from enum import Enum

class BeatMapDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

class HitObject_Type(Enum):
    Hit_Circle = 0
    Slider = 1
    New_Combo = 2
    Spinner = 3
    Combo_Skips4 = 4
    Combo_Skips5 = 5
    Combo_Skips6 = 6
    Hold_Note = 7
    Unknown = -1

class HitObject():
    def __init__(self, x, y, time, type, hitsound,  objectParams, hitSample) -> None:
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.hitsound = hitsound
        self.objectParams = objectParams
        self.hitSample = hitSample
        self.type_name = self.__gettypename__()

    def __gettypename__(self) -> HitObject_Type:
        return HitObject_Type(int(self.type))
    #TODO parser currently not completed
    @classmethod
    def from_str(cls, str:str):
        splitted = str.split(',')
        return cls(
            x=cls.__safe_get__(list=splitted, idx=0),
            y=cls.__safe_get__(list=splitted, idx=1),
            time=cls.__safe_get__(list=splitted, idx=2),
            type=cls.__safe_get__(list=splitted, idx=3, NullValue="-1"),
            hitsound=cls.__safe_get__(list=splitted, idx=4),
            objectParams=cls.__safe_get__(list=splitted, idx=5),
            hitSample=cls.__safe_get__(list=splitted, idx=6)
        )
    
    @classmethod
    def __safe_get__(cls, list:list, idx:int, NullValue=""):
        if len(list) >= idx +1:
            return list[idx]
        else:
            return NullValue
        

def str_to_HitObject(str:str):
    tmp = str.split(',')

lines = [""]
with open("Maps/839864 S3RL - Catchit (Radio Edit)/S3RL - Catchit (Radio Edit) (Rolniczy) [Ex].osu", 'r') as f:
    lines.clear()
    lines = f.readlines()
try:
    HO_idx = lines.index("[HitObjects]\n") + 1
except ValueError:
    exit("HitObject not found")
#print(f"starts: {HO_idx} - ends: {len(lines)}")
hitObjects:list[HitObject] = []
hitObjects.clear()
for idx in range(HO_idx, len(lines)):
    obj = HitObject.from_str(lines[idx])
    #if obj.type == 0:
    hitObjects.append(obj)

for obj in hitObjects:
    print(f"idx: {hitObjects.index(obj)} | Type: {obj.type_name.name}")