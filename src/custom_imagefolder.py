import os
from pathlib import Path
from typing import Set
from torchvision import datasets

VALID_EXTS: Set[str] = {".jpg",".jpeg",".png",".bmp",".webp",".tif",".tiff"}
EXCLUDE_DIRS: Set[str] = {"splits",".ipynb_checkpoints"}

def _has_images(p: Path) -> bool:
    if not p.is_dir(): return False
    for f in p.iterdir():
        if f.is_file() and f.suffix.lower() in VALID_EXTS:
            return True
    return False

class SafeImageFolder(datasets.ImageFolder):
    """
    ImageFolder that ignores known non-class folders (e.g., 'splits') and
    any directory without at least one valid image. Fixes 'Found no valid file
    for the classes splits' error.
    """
    def find_classes(self, directory: str):
        directory = Path(directory)
        classes = []
        for d in sorted(os.listdir(directory)):
            if d.startswith(".") or d.startswith("_"):  # ignore hidden/underscore
                continue
            if d.lower() in EXCLUDE_DIRS:
                continue
            p = directory / d
            if _has_images(p):
                classes.append(d)
        if not classes:
            raise FileNotFoundError(f"No valid class folders with images under: {directory}")
        class_to_idx = {c: i for i, c in enumerate(classes)}
        return classes, class_to_idx
