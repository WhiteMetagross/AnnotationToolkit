#This initializes the modules package and imports specific functions from various submodules.

from .detector import detect_image
from .tracker import track_video
from .annotator import manual_annotate
from .behavior import tag_behaviors
from .exporter import merge_annotations

__all__ = [
    'detect_image',
    'track_video', 
    'manual_annotate',
    'tag_behaviors',
    'merge_annotations'
]