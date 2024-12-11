import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from enum import Enum
import pathlib
import re

class BeatMapDataSet(Dataset):
    def __init__(self) -> None:
        super().__init__()

class HitObject_Type(Enum):
    Hit_Circle = 1
    Slider = 2
    Spinner = 8
    unknown = -1

class color_flag_enum(Enum):
    none = 0
    new_combo = 4
    one_skip = 16
    two_skip = 32
    three_skip = 64
    unknown = -1

class HitObject():
    def __init__(self, x="", y="", time="", type="", hitsound="",  objectParams="", hitSample="", is_empty=False) -> None:
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.hitsound = hitsound
        self.objectParams = objectParams
        self.hitSample = hitSample
        flag, type_name = self.__gettypename__()
        self.type_name = type_name
        self.color_flag = flag
        self.is_empty:bool = is_empty

    def __gettypename__(self) -> tuple[color_flag_enum, HitObject_Type]:
        itype = int(self.type)
        flag = color_flag_enum.unknown
        t = HitObject_Type.unknown
        flags = [4,16,32,64]
        for f in flags:
            if itype - f > 0:
                flag = color_flag_enum(f)
                t = HitObject_Type(itype - f)
                break
        if flag == color_flag_enum.unknown and t == HitObject_Type.unknown:
            flag = color_flag_enum.none
            t = HitObject_Type(itype)
        return flag, t


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
            hitSample=cls.__safe_get__(list=splitted, idx=6),
            is_empty=False
        )
    
    @classmethod
    def __safe_get__(cls, list:list[str], idx:int, NullValue=""):
        if len(list) >= idx +1:
            list[idx]
            return list[idx]
        else:
            return NullValue

class BeatMap:
    '''
    Represents a BeatMap Object containing all Information provided by the files
    
    Parameters
        BeatMap_Dir -> the relative path to the BeatMap Directory
    '''
    def __init__(self, BeatMap_Dir:str, Filter:list[HitObject_Type]=[]) ->None:
        self.BMDir = BeatMap_Dir
        '''relative BeatMap Directory'''
        self.Filter = Filter
        '''Filter option for specific HitObjects'''
        self.Audio_Path = pathlib.Path(BeatMap_Dir).joinpath("audio.mp3") if pathlib.Path(BeatMap_Dir).joinpath("audio.mp3").exists() and pathlib.Path(BeatMap_Dir).joinpath("audio.mp3").is_file() else pathlib.Path()
        '''Path of the Audio File | Beatmap Path if it wasn't found'''
        self.HitObjects:dict[str, list[HitObject]] = self.__getHitObjects__()
        '''HitObjects mapped to the difficulty'''
        

    def __getHitObjects__(self) -> dict[str, list[HitObject]]:
        HODic:dict[str, list[HitObject]] = {}
        HODic.clear()
        for child in pathlib.Path(self.BMDir).glob('*.osu'):
            #extract difficulty name as key in dict
            difficulty = re.search('\\[[^\\]]*\\]', child.name)
            if difficulty:
                difficulty = difficulty.group()
            else:
                difficulty = "not found"
            
            #retrieve hitobjects
            with child.open('r') as f:
                lines = f.readlines()
                try:
                    HO_idx = lines.index("[HitObjects]\n") + 1
                except ValueError:
                    HODic[difficulty] = [HitObject(is_empty=True)]
                    continue
                HODic[difficulty] = [HitObject.from_str(lines[idx]) for idx in range(HO_idx, len(lines))]
                if len(self.Filter) > 0:
                    HODic[difficulty] = [ho for ho in HODic[difficulty] if ho.type_name in self.Filter]

        return HODic

        


# lines = [""]
# with open("Maps/839864 S3RL - Catchit (Radio Edit)/S3RL - Catchit (Radio Edit) (Rolniczy) [Ex].osu", 'r') as f:
#     lines.clear()
#     lines = f.readlines()
# try:
#     HO_idx = lines.index("[HitObjects]\n") + 1
# except ValueError:
#     exit("HitObject not found")
# #print(f"starts: {HO_idx} - ends: {len(lines)}")
# hitObjects:list[HitObject] = []
# hitObjects.clear()
# for idx in range(HO_idx, len(lines)):
#     obj = HitObject.from_str(lines[idx])
#     #if obj.type == 0:
#     hitObjects.append(obj)

# for obj in hitObjects:
#     print(f"idx: {hitObjects.index(obj)} | Type: {obj.type_name.name} | Combo: {obj.color_flag}")

# for child in pathlib.Path("Maps/839864 S3RL - Catchit (Radio Edit)").glob('*.osu'):
#     print(child.name)
#     dif = re.search('\\[[^\\]]*\\]', child.name)
#     if dif:
#         print(dif.group())

bm = BeatMap("Maps/839864 S3RL - Catchit (Radio Edit)", Filter=[HitObject_Type.Hit_Circle])
for ho in bm.HitObjects:
    print(f"{ho} object count: {len(bm.HitObjects[ho])}")
print(f"Audio File: {bm.Audio_Path.name}")