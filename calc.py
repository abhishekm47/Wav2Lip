import os
from pathlib import Path
from glob import glob
import re

dataset_path = "/home/ubuntu/Wav2Lip/dataset"



def _find_filenames(file_dir_path, file_pattern): return list(file_dir_path.glob(file_pattern))


data = glob(dataset_path+"/*/")

total_vid_duration = []

for filePath in data:

    folderName = os.path.basename(os.path.normpath(filePath))

    video_id = folderName[:12]
    
    hashed_video_id = str(abs(hash(video_id)) % (10 ** 8))
    
    #print("normal_video_id: {}".format(video_id))
    #print("hashed_video_id: {}".format(hashed_video_id))
    last_two_digits_of_hashed_video_id = int(hashed_video_id[len(hashed_video_id) - 2: len(hashed_video_id)])
    
    if last_two_digits_of_hashed_video_id < 10:
        new_video_id = hashed_video_id+folderName[-22:]
        
        
            
        trajectory_dir = filePath+'trajectories/'
        
        trajectories = glob(trajectory_dir+"/*/")
        
        if trajectories == []:
            print("no trajectories at '{}'".format(folderName))
            print("skipping")
            continue
        
        
        for trajectory in trajectories:
            
            trajectory_name = os.path.basename(os.path.normpath(trajectory))
            
            align_frames_dir = trajectory+'align'
            
            all_valid_frames = _find_filenames(Path(align_frames_dir), '*.jpg')
            all_video_frames = [str(filename) for filename in all_valid_frames]
            all_video_frames.sort(key=lambda f: int(re.sub('\D', '', f)))
            duration_of_trajectory = len(all_video_frames)/25
            total_vid_duration.append(duration_of_trajectory)
            print("trajectory_name:{}, total duration: {} s".format(trajectory_name, duration_of_trajectory))
            


import statistics
avg_video_duration = statistics.mean(total_vid_duration)

print("AVG. video duration in AVSpeech dataset: {}".format(avg_video_duration))