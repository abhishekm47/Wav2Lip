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

def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))


data = glob(dataset_path+"/*/")


for filePath in data:

    folderName = os.path.basename(os.path.normpath(filePath))

    video_id = folderName[:12]
    
    hashed_video_id = str(abs(hash(video_id)) % (10 ** 8))
    
    #print("normal_video_id: {}".format(video_id))
    #print("hashed_video_id: {}".format(hashed_video_id))
    
    last_two_digits_of_hashed_video_id = int(hashed_video_id[len(hashed_video_id) - 2: len(hashed_video_id)])
    
    if last_two_digits_of_hashed_video_id < 10:
        new_video_id = hashed_video_id+folderName[-22:]
        
        file_dir = val_set+new_video_id
            
        trajectory_dir = filePath+'trajectories/'
        
        trajectories = glob(trajectory_dir+"/*/")
        
        if trajectories == []:
            print("no trajectories at '{}'".format(folderName))
            print("skipping")
            continue
        
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, 0o777)
        
        for trajectory in trajectories:
            
            trajectory_name = os.path.basename(os.path.normpath(trajectory))
            
            align_frames_dir = trajectory+'align'
            audio_file = trajectory+'audio.wav'
            
            all_valid_frames = _find_filenames(Path(align_frames_dir), '*.jpg')
            all_video_frames = [str(filename) for filename in all_valid_frames]
            all_video_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
            
            new_trajectory_video_dir = os.path.join(file_dir, trajectory_name)
            
            if not os.path.exists(new_trajectory_video_dir):
                os.makedirs(new_trajectory_video_dir, 0o777)
            
            #copy frames 
            for frame in all_video_frames:
                shutil.copy2(frame, os.path.join(new_trajectory_video_dir, os.path.basename(frame)))
            
            #copy audio file    
            shutil.copy2(audio_file, os.path.join(new_trajectory_video_dir, 'audio.wav'))
    
    
    else:
        new_video_id = hashed_video_id+folderName[-22:]
        
        file_dir = train_set+new_video_id    
            
        trajectory_dir = filePath+'trajectories/'
        
        trajectories = glob(trajectory_dir+"/*/")
        
        if trajectories == []:
            print("no trajectories at '{}'".format(folderName))
            print("skipping")
            continue
        
        if not os.path.exists(file_dir):
            os.makedirs(file_dir, 0o777)
        
        for trajectory in trajectories:
            
            trajectory_name = os.path.basename(os.path.normpath(trajectory))
            
            align_frames_dir = trajectory+'align'
            audio_file = trajectory+'audio.wav'
            
            all_valid_frames = _find_filenames(Path(align_frames_dir), '*.jpg')
            all_video_frames = [str(filename) for filename in all_valid_frames]
            all_video_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
            
            new_trajectory_video_dir = os.path.join(file_dir, trajectory_name)
            
            if not os.path.exists(new_trajectory_video_dir):
                os.makedirs(new_trajectory_video_dir, 0o777)
            
            #copy frames 
            for frame in all_video_frames:
                shutil.copy2(frame, os.path.join(new_trajectory_video_dir, os.path.basename(frame)))
            
            #copy audio file    
            shutil.copy2(audio_file, os.path.join(new_trajectory_video_dir, 'audio.wav'))
            
                
            
        
        
        
        


