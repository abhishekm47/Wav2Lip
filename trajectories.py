import moviepy.editor as mp

from pydub import AudioSegment
import subprocess
#checks if GPU is available or not 
device_name = 'device_name'
fa_device_name = 'cpu'
is_cpu = True
import json

minimal_video_size = 300
sample_rate = 1
camera_change_threshold = 1
intensity_change_threshold = 1.5
max_frames=1024
increase_ = 0.05
max_crops = 1000
min_frames = 128
min_size = 256
output_size = 512
image_shape = None



#face Alignment library initialization
import face_alignment


fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False, device="cpu", face_detector='sfd')



import os

import cv2
from skimage.transform import resize
from skimage.color import rgb2gray


import imageio
import tqdm
import numpy as np

import matplotlib


def calculate_landmarks(video):
    
    results = {}
    for i, frame in enumerate(video):
        print('calculating FL pred for frame IDX: {}'.format(i))
        pred = fa.get_landmarks(frame)
        pred = [p.tolist() for p in pred]
        results[i] = pred
    return results

def load_landmarks_json(filePath):
    with open(filePath) as json_file:
        data = json.load(json_file)
#         result = [data[key] for key in sorted(data.keys(), reverse=False)]
        return data

def save_landmarks(landmarks, save_name):
    with open(save_name, 'w') as outfile:
        json.dump(landmarks, outfile)


def crop_face_from_frame_with_bbox(frame, bbox):

    increase_area = 0.10
    left, top, right, bot  = bbox
    width = right - left
    height = bot - top
    frame_shape = frame.shape

    
    #Computing aspect preserving bbox
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))
    
    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)
    h, w = bot - top, right - left
    
    top2, bot2, left2, right2 = max(0, top), min(bot, frame_shape[0]), max(0, left), min(right, frame_shape[1])
    crop_img = frame[top2:bot2, left2:right2]
    
    top_border = abs(top2 - top)
    bot_border = abs(bot2 - bot)
    left_border = abs(left2 - left)
    right_border = abs(right2 - right)
    
    crop_img = cv2.copyMakeBorder(crop_img, top=top_border, bottom=bot_border, left=left_border, right=right_border, borderType= cv2.BORDER_CONSTANT, value=[0,0,0] )
    crop_img = cv2.resize(crop_img, (output_size, output_size))
    
    #crop_img = cv2.flip( crop_img, 0 ) 
    crop_img = cv2.flip( crop_img, 1 )
    
    return crop_img


def compute_aspect_preserved_bbox(bbox, increase_area):
    left, top, right, bot = bbox
    width = right - left
    height = bot - top
 
    width_increase = max(increase_area, ((1 + 2 * increase_area) * height - width) / (2 * width))
    height_increase = max(increase_area, ((1 + 2 * increase_area) * width - height) / (2 * height))

    left = int(left - width_increase * width)
    top = int(top - height_increase * height)
    right = int(right + width_increase * width)
    bot = int(bot + height_increase * height)

    return (left, top, right, bot)



def one_box_inside_other(boxA, boxB):
    xA = boxA[0] <= boxB[0]
    yA = boxA[1] <= boxB[1]
    xB = boxA[2] >= boxB[2]
    yB = boxA[3] >= boxB[3]
    return xA and yA and xB and yB



def get_bounding_box_from_landmarks(pts):
    min_x, min_y = np.min(pts, axis=0)
    max_x, max_y = np.max(pts, axis=0)
    return (min_x, min_y, max_x, max_y)

    
def bb_intersection_over_union(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou

def check_camera_motion(current_frame, previous_frame):
    flow = cv2.calcOpticalFlowFarneback(previous_frame, current_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.quantile(mag, [0.25, 0.5, 0.75], overwrite_input=True)



def store(video_path, trajectories, end, chunks_data, fps, landmarks, audio):
    print("stroring trajectory....")
    for i, (start, frame_list) in enumerate(trajectories):
    
        if(end-start < 0):
        	print("Too few frames, not saving")
        	continue
        
        fileName = os.path.basename(video_path).split('.')[0]
        
        if not os.path.exists(fileName):
            os.makedirs(fileName, 0o777)

        video_id = fileName+'/'+fileName

        landmarks_save_name = fileName+'/landmarks.json'
        #save landmarks if file didn't exists
        if not os.path.exists(landmarks_save_name):
            
            save_landmarks(landmarks, landmarks_save_name)

        name = (video_id + "#" + str(start).zfill(6) + "#" + str(end).zfill(6) + ".mp4")
        audioName = (video_id + "#" + str(start).zfill(6) + "#" + str(end).zfill(6) + ".wav")
        finaleFileName = (video_id + "#" + str(start).zfill(6) + "#" + str(end).zfill(6) + "#"+str(frame_list[0][1][0])+"X"+str(frame_list[0][1][1]) + ".mp4")
        chunks_data.append({ 'start': start, 'end': end, 'fps': fps,
                            'video_id': video_id, 'height': frame_list[0][0].shape[0], 'width': frame_list[0][0].shape[1]})

        soundFile = audio[int((start/fps)*1000):int((end/fps)*1000)]
        soundFile.export(audioName, format="wav")
        # crop_imgs = []
        videox = cv2.VideoWriter(name, cv2.VideoWriter_fourcc(*'mp4v'), fps, (output_size,output_size))

        for i, (frame, bbox) in enumerate(frame_list):
            this_crop = crop_face_from_frame_with_bbox(frame, bbox)
            # crop_imgs.append(this_crop)
            videox.write(this_crop)
        
        videox.release()

        #stich audio to clip 
        # subprocess.call(['ffmpeg', '-i', name, '-i', audioName, '-c', 'copy', '-map', '0:v:0', '-map', '1:a:0', '-shortest', finaleFileName])
        subprocess.call(['ffmpeg', '-i', name, '-i', audioName, '-map', '0:v:0', '-map', '1:a:0', '-shortest', finaleFileName])





def process_video(video_path, landmarks_json_path=None):
    video = imageio.get_reader(video_path)
    subprocess.call(['ffmpeg', '-i', video_path, '-q:a', '0', '-map', 'a', 'temp.mp3'])
    audio = AudioSegment.from_mp3("temp.mp3")
    fps = video.get_meta_data()['fps']
    trajectories = []
    previous_frame = None
    chunks_data = []


    
    if landmarks_json_path == None:
        landmarks = calculate_landmarks(video)
        print("calculating landmarks ...")
    else:
        landmarks = load_landmarks_json(landmarks_json_path)
        print("loading landmarks ...")
        print(landmarks_json_path)
    try:
        print("processing_frames ...")
        for i, frame in enumerate(video):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if minimal_video_size > min(frame.shape[0], frame.shape[1]):
                return chunks_data
            if i % sample_rate != 0:
                continue
            pred = landmarks[str(i)]
            bboxes = []
            for p in pred:
                bbox = get_bounding_box_from_landmarks(p)
                bboxes.append(bbox)
                print(bbox)
            
            
            bboxes_valid = bboxes
            

            ### Check if frame is valid
            if previous_frame is None:
                previous_frame = rgb2gray(
                    resize(frame, (256, 256), preserve_range=True, anti_aliasing=True, mode='constant'))

                current_frame = previous_frame
                previous_intensity = np.median(frame.reshape((-1, frame.shape[-1])), axis=0)
                current_intensity = previous_intensity
            else:
                current_frame = rgb2gray(
                    resize(frame, (256, 256), preserve_range=True, anti_aliasing=True, mode='constant'))
                current_intensity = np.median(frame.reshape((-1, frame.shape[-1])), axis=0)

            flow_quantiles = check_camera_motion(current_frame, previous_frame)
            camera_criterion = flow_quantiles[1] > camera_change_threshold
            previous_frame = current_frame
            intensity_criterion = np.max(np.abs(previous_intensity - current_intensity)) > intensity_change_threshold
            print(previous_intensity,  current_intensity)
            previous_intensity = current_intensity
            no_person_criterion = len(bboxes) < 0
            criterion = no_person_criterion or camera_criterion or intensity_criterion
            print(i)
            print("CRITERION")
            print(criterion)
            print(flow_quantiles)
            print(np.max(np.abs(previous_intensity - current_intensity)))
            

            if criterion:
                store(video_path, trajectories, i, chunks_data, fps, landmarks, audio)
                trajectories = []

            ## For each trajectory check the criterion
            not_valid_trajectories = []
            valid_trajectories = []

            for trajectory in trajectories:
                
                # for frame_obj in trajectory[1]:
                tube_bbox = compute_aspect_preserved_bbox(trajectory[1][-1][1], 0.10)
                number_of_intersections = 0
                current_bbox = None
                for bbox in bboxes_valid:
                    intersect = bb_intersection_over_union(tube_bbox, bbox) > 0
                    if intersect:
                        current_bbox = bbox


                if current_bbox is None:
                    not_valid_trajectories.append(trajectory)
                    continue

                if number_of_intersections > 1:
                    not_valid_trajectories.append(trajectory)
                    continue

                if not one_box_inside_other(tube_bbox, current_bbox):
                    not_valid_trajectories.append(trajectory)
                    continue

                # if len(trajectory[1]) >= max_frames:
                #     not_valid_trajectories.append(trajectory)
                #     continue

                valid_trajectories.append(trajectory)

            store(video_path, not_valid_trajectories, i, chunks_data, fps, landmarks, audio)
            trajectories = valid_trajectories

            ## Assign bbox to trajectories, create new trajectories
            for bbox in bboxes_valid:
                intersect = False
                for trajectory in trajectories:
                    
                    # for frame_obj in trajectory[1]:
                    tube_bbox = compute_aspect_preserved_bbox(trajectory[1][-1][1], 0.10)
                    intersect = bb_intersection_over_union(tube_bbox, bbox) > 0
                    if intersect:
                        #trajectory[1] = join(tube_bbox, bbox)
                        trajectory[1].append((frame, bbox))
                        break

                ## Create new trajectory
                if not intersect:
                    trajectories.append([ i, [(frame, bbox)]])

            if len(chunks_data) > max_crops:
                break

    except IndexError:
        None

    store(video_path, trajectories, i + 1, chunks_data, fps, landmarks, audio)
    return chunks_data
    
process_video('TonightShowXYSelenaGomez.mp4', landmarks_json_path='/Users/rijulgupta/Desktop/AnonymousAI/universal-translator/TonightShowXYSelenaGomez/landmarks.json')