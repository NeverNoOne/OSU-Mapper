'''osu wrapper

This library contains all classes needed to read an osu beatmapset

'''
from enum import Enum
import pathlib
import inspect
from audio.analysis import AudioAnalyser

class General:
    def __init__(self) -> None:
        self.AudioFilename:str = ""
        self.AudioLeadIn:int = 0
        self.PreviewTime:int = -1
        self.Countdown:int = 1
        self.SampleSet:str = "Normal"
        self.StackLeniency:float = 0.7
        self.mode:Mode_Type = Mode_Type.osu
        self.LetterboxInBreaks:bool = False
        self.UseSkinSprites:bool = False
        self.OverlayPosition:str = "NoChange"
        self.SkinPreference:str = ""
        self.EpilepsyWarning:bool = False
        self.CountdownOffset:int = 0
        self.SpecialStyle:bool = False
        self.WidescreenStoryboard:bool = False
        self.SamplesMatchPlaybackRate:bool = False
        self.AudioFilePath:str = ""
        self.AudioAnalyser:AudioAnalyser

class Metadata:
    def __init__(self) -> None:
        self.Title:str = ""
        self.TitleUnicode:str = ""
        self.Artist:str = ""
        self.ArtistUnicode:str = ""
        self.Creator:str = ""
        self.Version:str = ""
        self.Source:str = ""
        self.Tags:list[str] = []
        self.BeatmapID:int = 0
        self.BeatmapSetID:int = 0

class Difficulty:
    def __init__(self) -> None:
        self.HPDrainRate:float = 0
        self.CircleSize:float = 0
        self.OverallDifficulty:float = 0
        self.ApproachRate:float = 0
        self.SliderMultiplier:float = 0
        self.SliderTickRate:float = 0
        
class Mode_Type(Enum):
    osu = 0
    taiko = 1
    catch = 2
    mania = 3

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
    #TODO color flags are not correct rn
    def __gettypename__(self) -> tuple[color_flag_enum, HitObject_Type]:
        itype = int(self.type)
        flag = color_flag_enum.unknown
        t = HitObject_Type.unknown
        flags = [64,32,16,4]
        for f in flags:
            if itype - f > 0:
                flag = color_flag_enum(f)
                if itype - f not in [item.value for item in HitObject_Type]:
                    t = HitObject_Type(itype - f - 4)
                else:
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
        BeatMapFile -> the relative path to the BeatMap File
    '''
    def __init__(self, BeatMap_File:str, Filter:list[HitObject_Type]=[]) ->None:
        self.BMFile = BeatMap_File
        '''relative of the Beatmap File'''
        self.General:General = self.__getGeneral__()
        '''contains General settings of the Beatmap'''
        self.Metadata:Metadata = self.__getMetadata__()
        '''contains metadata about the Beatmap'''
        self.Difficulty:Difficulty = self.__getDifficulty__()
        '''contains the difficulty settings of the Beatmap'''
        self.Filter = Filter
        '''Filter option for specific HitObjects'''
        self.HitObjects:list[HitObject] = self.__getHitObjects__()
        '''HitObjects of the current BeatMap'''
        #self.AudioFilePath:str = pathlib.Path(BeatMap_File).parent
    
    def __getGeneral__(self) -> General:
        Gn:General = General()
        with open(self.BMFile, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            Start = lines.index('[General]\n')
            End = lines.index('\n',Start)
            for idx in range(Start + 1,End):
                option, value = lines[idx].split(':', 1)
                value = value.removesuffix('\n').removeprefix(' ')
                match option:
                    case "AudioFilename":
                        Gn.AudioFilename = value
                        Gn.AudioFilePath = str(pathlib.Path(self.BMFile).parent.joinpath(value))
                        Gn.AudioAnalyser = AudioAnalyser(Gn.AudioFilePath)
                    case "AudioLeadIn":
                        Gn.AudioLeadIn = int(value)
                    case "PreviewTime":
                        Gn.PreviewTime = int(value)
                    case "Countdown":
                        Gn.Countdown = int(value)
                    case "SampleSet":
                        Gn.SampleSet = value
                    case "StackLeniency":
                        Gn.StackLeniency = float(value)
                    case "Mode":
                        Gn.mode = Mode_Type(int(value))
                    case "LetterboxInBreaks":
                        Gn.LetterboxInBreaks = bool(value)
                    case "UseSkinSprites":
                        Gn.UseSkinSprites = bool(value)
                    case "OverlayPosition":
                        Gn.OverlayPosition = value
                    case "SkinPreference":
                        Gn.SkinPreference = value
                    case "EpilepsyWarning":
                        Gn.EpilepsyWarning = bool(value)
                    case "CountdownOffset":
                        Gn.CountdownOffset = int(value)
                    case "SpecialStyle":
                        Gn.SpecialStyle = bool(value)
                    case "WidescreenStoryboard":
                        Gn.WidescreenStoryboard = bool(value)
                    case "SamplesMatchPlaybackRate":
                        Gn.SamplesMatchPlaybackRate = bool(value)
        return Gn

    def __getMetadata__(self) -> Metadata:
        md = Metadata()
        with open(self.BMFile, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            Start = lines.index('[Metadata]\n')
            End = lines.index('\n',Start)
            for idx in range(Start + 1,End):
                option, value = lines[idx].split(':', 1)
                value = value.removesuffix('\n')
                match option:
                    case "Title":
                        md.Title = value
                    case "TitleUnicode":
                        md.TitleUnicode = value
                    case "Artist":
                        md.Artist = value
                    case "ArtistUnicode":
                        md.ArtistUnicode = value
                    case "Creator":
                        md.Creator = value
                    case "Version":
                        md.Version = value
                    case "Source":
                        md.Source = value
                    case "Tags":
                        md.Tags = value.split(' ')
                    case "BeatmapID":
                        md.BeatmapID = int(value)
                    case "BeatmapSetID":
                        md.BeatmapSetID = int(value)
        return md

    def __getDifficulty__(self) -> Difficulty:
        dif = Difficulty()
        with open(self.BMFile, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            Start = lines.index('[Difficulty]\n')
            End = lines.index('\n',Start)
            for idx in range(Start + 1,End):
                option, value = lines[idx].split(':', 1)
                value = value.removesuffix('\n')
                match option:
                    case "HPDrainRate":
                        dif.HPDrainRate = float(value)
                    case "CircleSize":
                        dif.CircleSize = float(value)
                    case "OverallDifficulty":
                        dif.OverallDifficulty = float(value)
                    case "ApproachRate":
                        dif.ApproachRate = float(value)
                    case "SliderMultiplier":
                        dif.SliderMultiplier = float(value)
                    case "SliderTickRate":
                        dif.SliderTickRate = float(value)
        return dif

    def __getHitObjects__(self) -> list[HitObject]:
        #retrieve hitobjects
        with open(self.BMFile, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
            try:
                HO_idx = lines.index("[HitObjects]\n") + 1
            except ValueError:
                print(f"\033 [93m no hitobjects found in {self.BMFile} \033[0m")
                return [HitObject(is_empty=True)]
            HitObjects = [HitObject.from_str(lines[idx]) for idx in range(HO_idx, len(lines))]
            if len(self.Filter) > 0:
                HitObjects = [ho for ho in HitObjects if ho.type_name in self.Filter]
        return HitObjects

    @classmethod
    def getMaps_from_MapDir(cls, Dir:str):
        '''
        returns a list of Beatmaps from the given Directory
        '''
        path = pathlib.Path(Dir)
        
        return [cls(str(child)) for child in path.glob('*.osu')]
    
    @classmethod
    def getMaps_from_Dir(cls, Dir:str):
        #return [cls.getMaps_from_MapDir(str(mapdir)) for mapdir in pathlib.Path(Dir).iterdir() if mapdir.is_dir]
        r:list[BeatMap] = []
        r.clear()
        for mapdir in pathlib.Path(Dir).iterdir():
            if mapdir.is_dir():
                r.extend(cls.getMaps_from_MapDir(str(mapdir)))
        return r
import tracemalloc
tracemalloc.start()
p = BeatMap.getMaps_from_Dir("Maps")
current, peak = tracemalloc.get_traced_memory()
print(f"{current / 1024 / 1024:.2f}")
print(f"{peak / 1024 / 1024:.2f}")
tracemalloc.stop()