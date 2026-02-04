import os
import librosa
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

def extract_mfcc(path, n_mfcc=20, max_len=100):
    y, sr = librosa.load(path, sr=22050)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len-mfcc.shape[1])))
    else:
        mfcc = mfcc[:,:max_len]
    return mfcc.astype(np.float32)

def load_local_dataset(base_path):
    X, y = [], []
    for label, cls in enumerate(["REAL","FAKE"]):
        folder = os.path.join(base_path, cls)
        for f in os.listdir(folder):
            mfcc = extract_mfcc(os.path.join(folder,f))
            X.append(mfcc)
            y.append(label)
    dataset = TensorDataset(torch.tensor(X), torch.tensor(y))
    return DataLoader(dataset, batch_size=16, shuffle=True)
