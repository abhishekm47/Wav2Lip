from os import listdir, path
import numpy as np
import scipy
import cv2
import os
import sys
import argparse
import Wav2Lip.audio as audio
import json
import subprocess
import random
import string
from tqdm import tqdm
from glob import glob
import torch
from pathlib import Path
import Wav2Lip.face_detection as face_detection
from Wav2Lip.face_crop import crop_face_from_frame_with_bbox
from Wav2Lip.models import Wav2Lip
import platform

checkpoint_path = "/home/ubuntu/making_with_ml/ai_dubs/Wav2Lip/checkpoints/wav2lip.pth"

static = False

final_fps = 25.

bbox_pads = [0, 10, 0, 0]

face_det_batch_size = 16

wav2lip_batch_size = 128

resize_factor = 1

crop = [0, -1, 0, -1]

submit_box = [-1, -1, -1, -1]

rotate = False

nosmooth = False

final_img_size = 96

face_data = []


def get_smoothened_boxes(boxes, T):
	for i in range(len(boxes)):
		if i + T > len(boxes):
			window = boxes[len(boxes) - T:]
		else:
			window = boxes[i: i + T]
		boxes[i] = np.mean(window, axis=0)
	return boxes


def face_detect(images):
	detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D,
											flip_input=False)

	batch_size = face_det_batch_size

	while 1:
		predictions = []
		try:
			for i in tqdm(range(0, len(images), batch_size)):
				frames_batch = [obj['frame'] for obj in images[i:i + batch_size]]
				predictions.extend(detector.get_detections_for_batch(np.array(frames_batch)))
		except RuntimeError:
			if batch_size == 1:
				raise RuntimeError(
					'Image too big to run face detection on GPU. Please use the --resize_factor argument')
			batch_size //= 2
			print('Recovering from OOM error; New batch size: {}'.format(batch_size))
			continue
		break

	results = []
	pady1, pady2, padx1, padx2 = bbox_pads
	for rect, image in zip(predictions, images):
		if rect is None:
			# check this frame where the face was not detected.
			cv2.imwrite(
				'/home/ubuntu/making_with_ml/ai_dubs/Wav2Lip/temp/faulty_frame.jpg', image['frame'])
			raise ValueError(
				'Face not detected! Ensure the video contains a face in all the frames.')

		y1 = max(0, rect[1] - pady1)
		y2 = min(image['frame'].shape[0], rect[3] + pady2)
		x1 = max(0, rect[0] - padx1)
		x2 = min(image['frame'].shape[1], rect[2] + padx2)

		results.append([x1, y1, x2, y2])

	boxes = np.array(results)
	if not nosmooth: boxes = get_smoothened_boxes(boxes, T=5)
	results = [[image, (y1, y2, x1, x2)]
						for image, (x1, y1, x2, y2) in zip(images, boxes)]

	del detector
	return results


def datagen(face_det_results, frames, mels):
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	for i, m in enumerate(mels):
		idx = 0 if static else i % len(frames)
		frame_to_save = frames[idx]['frame'].copy()
		face_obj, coords = face_det_results[idx].copy()
  
		y1, y2, x1, x2 = coords
		face = face_obj['frame'][y1: y2, x1:x2]
		face = cv2.resize(face, (final_img_size, final_img_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append({'frame':frame_to_save, 'path':face_obj['path']})
		coords_batch.append(coords)

		
		if len(img_batch) >= wav2lip_batch_size:
			img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

			img_masked = img_batch.copy()
			img_masked[:, final_img_size//2:] = 0

			img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
			mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

			yield img_batch, mel_batch, frame_batch, coords_batch
			img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

	if len(img_batch) > 0:
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, final_img_size//2:] = 0

		img_batch = np.concatenate((img_masked, img_batch), axis=3) / 255.
		mel_batch = np.reshape(
			mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch


mel_step_size = 16
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} for inference.'.format(device))


def _load(checkpoint_path):
	if device == 'cuda':
		checkpoint = torch.load(checkpoint_path)
	else:
		checkpoint = torch.load(checkpoint_path,
								map_location=lambda storage, loc: storage)
	return checkpoint


def load_model(path):
	model = Wav2Lip()
	print("Load checkpoint from: {}".format(path))
	checkpoint = _load(path)
	s = checkpoint["state_dict"]
	new_s = {}
	for k, v in s.items():
		new_s[k.replace('module.', '')] = v
	model.load_state_dict(new_s)

	model = model.to(device)
	return model.eval()


def _find_filenames(file_dir_path, file_pattern): return list(
	file_dir_path.glob(file_pattern))


def wav2lip_lip_frame(face_file, audio_file, combined_frame_dir, crop_frames_dir, original_crop_frames_dir):
	if os.path.isfile(face_file):
		full_frames = [cv2.imread(face_file)]
		fps = (final_fps)
		
	else:
		frames_paths = _find_filenames(Path(face_file), '*.jpg')
		frames_paths = [str(filename) for filename in frames_paths]

		fps = final_fps

		print('Reading video frames...')

		full_frames = []

		for file_path in frames_paths:
			frame = cv2.imread(file_path)
			if resize_factor > 1:
				frame = cv2.resize(
					frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))
					
			if rotate:
				frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
			
			y1, y2, x1, x2 = crop
			if x2 == -1: x2 = frame.shape[1]
			if y2 == -1: y2 = frame.shape[0]
			
			frame = frame[y1:y2, x1:x2]
			
			full_frames.append({'frame':frame, 'path':file_path})
			
	print("Number of frames available for inference: "+str(len(full_frames)))
	
	if not audio_file.endswith('.wav'):
		print('Extracting raw audio...')
		command = 'ffmpeg -y -i {} -strict -2 {}'.format(audio_file, '/home/ubuntu/making_with_ml/ai_dubs/Wav2Lip/temp/temp.wav')
		
		subprocess.call(command, shell=True)
		audio_file = '/home/ubuntu/making_with_ml/ai_dubs/Wav2Lip/temp/temp.wav'
		
	wav = audio.load_wav(audio_file, 16000)
	mel = audio.melspectrogram(wav)
	print(mel.shape)
	
	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan! Using a TTS voice? Add a small epsilon noise to the wav file and try again')
	
	mel_chunks = []
	mel_idx_multiplier = 80./fps
	i = 0
	
	while 1:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			mel_chunks.append(mel[:, len(mel[0]) - mel_step_size:])
			break
		mel_chunks.append(mel[:, start_idx: start_idx + mel_step_size])
		i += 1
		
		
	print("Length of mel chunks: {}".format(len(mel_chunks)))
	
	full_frames = full_frames[:len(mel_chunks)]
	
	batch_size = wav2lip_batch_size
	
	if submit_box[0] == -1:
		if not static:
			#BGR2RGB for CNN face detection
			face_det_results = face_detect(full_frames.copy())
		else:
			face_det_results = face_detect([frames[0].copy()])
			
	else:
		print('Using the specified bounding box instead of face detection...')
		y1, y2, x1, x2 = submit_box
		face_det_results = [[f['frame'][y1: y2, x1:x2], (y1, y2, x1, x2)] for f in full_frames]
		
	gen = datagen(face_det_results.copy(), full_frames.copy(), mel_chunks)
	bboxes_face = []
	count = 0
	if not os.path.exists(combined_frame_dir):
		os.makedirs(combined_frame_dir, 0o777)
	
	combined_frame_save_str = combined_frame_dir + "/{}"
	
	if not os.path.exists(crop_frames_dir):
		os.makedirs(crop_frames_dir, 0o777)
		
	crop_frames_save_str = crop_frames_dir + "/{}"
	
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,
											total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = load_model(checkpoint_path)
			print("Model loaded")
			
			frame_h, frame_w = full_frames[0]['frame'].shape[:-1]
		
		img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
		
		mel_batch = torch.FloatTensor(
			np.transpose(mel_batch, (0, 3, 1, 2))).to(device)
		
		with torch.no_grad():
			pred = model(mel_batch, img_batch)
			
		pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
		
		
		
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			bboxes_face.append(f)
			p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
			
			f['frame'][y1:y2, x1:x2] = p
			this_crop = crop_face_from_frame_with_bbox(f['frame'], c)
			# save cropped frames output
			cv2.imwrite(crop_frames_save_str.format(os.path.basename(f['path'])), this_crop)
			# save combined frames output
			cv2.imwrite(combined_frame_save_str.format(os.path.basename(f['path'])), f['frame'])
			
			count=count+1
			
	if not os.path.exists(original_crop_frames_dir):
		os.makedirs(original_crop_frames_dir, 0o777)
		
	original_crop_frames_save_str = original_crop_frames_dir+ "/{}"
	
	
	for i, (f, bbox) in enumerate(face_det_results):
		this_crop = crop_face_from_frame_with_bbox(f['frame'], bbox)
		cv2.imwrite(original_crop_frames_save_str.format(os.path.basename(f['path'])), this_crop)
		
		
	torch.cuda.empty_cache()
	
	
	return {'original_frames': len(face_det_results),'Wav2Lip_frames': len(bboxes_face)}

	
