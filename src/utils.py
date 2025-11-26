import os
import cv2
import numpy as np
from tqdm import tqdm

def ensure_dirs():
    os.makedirs("outputs/models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)

def read_image(path, to_gray=True):
    img = cv2.imread(path)
    if img is None:
        return None
    if to_gray:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img
