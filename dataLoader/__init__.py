from .llff_video import LLFFVideoDataset, SSDDataset
from .brics import BRICSDataset

dataset_dict = {'ssd': SSDDataset, 'llffvideo':LLFFVideoDataset, 'brics':BRICSDataset}
