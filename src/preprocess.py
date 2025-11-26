import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from src.utils import read_image
from tqdm import tqdm

def preprocess_image(img, size=(150,50), equalize=True):
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if equalize:
        img = cv2.equalizeHist(img)
    return img

def load_from_folder(root_folder, size=(150,50), flatten=True):
    X = []
    y = []
    for label in sorted(os.listdir(root_folder)):
        label_dir = os.path.join(root_folder, label)
        if not os.path.isdir(label_dir):
            continue
        for fname in os.listdir(label_dir):
            fpath = os.path.join(label_dir, fname)
            img = read_image(fpath, to_gray=True)
            if img is None:
                continue
            img = preprocess_image(img, size=size)
            if flatten:
                X.append(img.flatten())
            else:
                X.append(img)
            y.append(label)
    X = np.array(X)
    y = np.array(y)
    return X, y

def load_from_csv(csv_path, base_path="", size=(150,50), flatten=True):
    df = pd.read_csv(csv_path)

    if 'filename' not in df.columns or 'plate' not in df.columns:
        raise ValueError("CSV deve conter as colunas: filename e plate.")

    X = []
    y = []

    for _, row in df.iterrows():
        img_path = os.path.join(base_path, row['filename']) if base_path else row['filename']

        img = read_image(img_path, to_gray=True)
        if img is None:
            print(f"[AVISO] Não foi possível carregar: {img_path}")
            continue

        img = preprocess_image(img, size=size)

        X.append(img.flatten() if flatten else img)
        y.append(row['plate'])   

    return np.array(X), np.array(y)


def encode_labels(y):
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    return y_enc, le
