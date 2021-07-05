import os
from pathlib import Path
from glob import glob
import subprocess
import shutil
import hashlib
import re

dataset_path = "/home/ubuntu/Wav2Lip/dataset"

save_dir = "/home/ubuntu/Wav2Lip/preprocessed_avspeech"

val_set = save_dir+'/val/'
train_set = save_dir+'/train/'

train_videos = glob(train_set+"/*/")
val_videos = glob(val_set+"/*/")

val_file = "/home/ubuntu/Wav2Lip/filelists/val.txt"
train_file = "/home/ubuntu/Wav2Lip/filelists/train.txt"

import sys

sys.stdout = open(train_file, 'wt', encoding="utf-8")

for video in train_videos:
    video_basename = os.path.basename(video)
    
    trajectories = glob(video+"/*/")
    for t in trajectories:
        print(t)
        
        
    