import logging
import os
import random
from dataclasses import dataclass

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import csv
from tqdm import tqdm

import json
from decord import VideoReader, cpu
import numpy as np
import torch
import utils
import time
import pandas as pd
import sys
from glob import glob
import cv2

def resize_frames(frames, target_height, target_width):
    if frames.shape[1] == target_height and frames.shape[2] == target_width:
        return frames
    
    resized_frames = np.zeros((frames.shape[0], target_height, target_width, frames.shape[3]), dtype=frames.dtype)
    for i in range(frames.shape[0]):
        resized_frames[i] = cv2.resize(frames[i], (target_width, target_height))
    return resized_frames


def load_video_inter(video_path, pos_window=None, clip_frames=16, target_height=None, target_width=None):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    if pos_window:
        start, end = int(pos_window[0]), int(pos_window[1])
        uniform_sampled_frames = np.linspace(start, end, clip_frames, dtype=int)
    else:
        uniform_sampled_frames = np.linspace(0, total_frame_num-1, clip_frames, dtype=int)
    frame_idx = uniform_sampled_frames.tolist()
    video = vr.get_batch(frame_idx).asnumpy()
    vr.seek(0)
    
    if target_height is not None and target_width is not None:
        video = resize_frames(video, target_height, target_width)
    
    return video

def load_video_intra(video_path, pos_window, clip_frames=16, neg_frames=100):
    vr = VideoReader(video_path, ctx=cpu(0))
    total_frame_num = len(vr)
    vr_fps = vr.get_avg_fps()
    video_time = total_frame_num / vr_fps

    pos_start, pos_end = int(pos_window[0]), int(pos_window[1])
    pos_end = min(pos_end, total_frame_num-1)
    uniform_sampled_frames = np.linspace(pos_start, pos_end, clip_frames, dtype=int)
    pos_frame_idxs = uniform_sampled_frames.tolist()

    neg_candidates = [i for i in range(total_frame_num) if i not in pos_frame_idxs]
    neg_frame_idxs = random.sample(neg_candidates, neg_frames)

    selected_frame_idxs = pos_frame_idxs + neg_frame_idxs
    selected_frame_idxs = sorted(selected_frame_idxs)

    sample_frames = neg_frames+clip_frames
    labels = torch.zeros(sample_frames, dtype=torch.bool)
    for i, frm_idx in enumerate(selected_frame_idxs):
        if frm_idx in pos_frame_idxs:
            labels[i] = 1

    frames = vr.get_batch(selected_frame_idxs).asnumpy() 
    vr.seek(0)
    return frames, selected_frame_idxs, video_time, labels


class ShortVideoDataset(Dataset):
    def __init__(
        self,
        processor,
        nextqa_anno_path,
        nextqa_video_path,
        qaego4d_anno_path,
        qaego4d_video_path, 
        clip_frames = 16,
    ):
        
        self.nextqa_video_path = nextqa_video_path
        self.qaego4d_video_path = qaego4d_video_path
        self.clip_frames = clip_frames
        self.processor = processor

        self.video_ids = []
        self.questions = []
        self.qids = []
        self.data_type = []
        self.time_references = []

        with open("/path/to/map_vid_vidorID.json") as file:
            self.vid_map = json.load(file)
        self.nextqa_data = pd.read_csv(nextqa_anno_path)
        for index, data in self.nextqa_data.iterrows():
            video_id = str(data['video'])

            self.video_ids.append(video_id)
            self.qids.append(data['qid'])
            self.questions.append(data['question'])
            self.time_references.append([0, data['frame_count']])

            self.data_type.append('nextqa')

 
        qaego4d_data = json.load(open(qaego4d_anno_path, 'r'))
        for data in qaego4d_data:
            self.video_ids.append(data['video_id'])
            self.qids.append(data['sample_id'])
            self.questions.append(data['question'])

            moment_start_frame = data['moment_start_frame']
            moment_end_frame = data['moment_end_frame']
            self.time_references.append([moment_start_frame, moment_end_frame])

            self.data_type.append('qaego4d')
            


    def __len__(self):
        """returns the length of dataframe"""
        return len(self.data_type)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        qid = str(self.qids[index])
        question = str(self.questions[index])
        data_type = self.data_type[index]
        time_reference = self.time_references[index]

        if data_type == 'nextqa':
            video_path = os.path.join(self.nextqa_video_path, f'{self.vid_map[video_id]}.mp4')
            video, _, _, _ = load_video_inter(video_path, clip_frames=self.clip_frames)

            outputs = self.processor(text=question, images=video, return_tensors="pt", padding='max_length', max_length=77)
            return {
                "video_ids": video_id,
                "qids": qid,
                "frames": outputs["pixel_values"],
                "questions": outputs['input_ids'][0],
                'attention_mask': outputs['attention_mask'][0],
            }  

        if data_type == 'qaego4d':
            video_path = os.path.join(self.qaego4d_video_path, f'{video_id}.mp4')
            video, _, _, _ = load_video_inter(video_path, pos_window=time_reference, clip_frames=self.clip_frames)

            
            outputs = self.processor(text=question, images=video, return_tensors="pt", padding='max_length', max_length=77)
            return {
                "video_ids": video_id,
                "qids": qid,
                "frames": outputs["pixel_values"],
                "questions": outputs['input_ids'][0][:77],
                'attention_mask': outputs['attention_mask'][0][:77],
            }  
            

class SYNLVideoDataset(Dataset):
    def __init__(
        self,
        processor,
        nextqa_anno_path,
        nextqa_video_path,
        qaego4d_anno_path,
        qaego4d_video_path, 
        clip_frames = 16,
        neg_frames = 100
    ):
        
        self.nextqa_video_path = nextqa_video_path
        self.qaego4d_video_path = qaego4d_video_path
        self.clip_frames = clip_frames
        self.neg_frames = neg_frames
        self.processor = processor

        self.video_ids = []
        self.questions = []
        self.qids = []
        self.data_type = []
        self.time_references = []

        with open("/path/to/map_vid_vidorID.json") as file:
            self.vid_map = json.load(file)
        self.nextqa_data = pd.read_csv(nextqa_anno_path)
        for index, data in self.nextqa_data.iterrows():
            video_id = str(data['video'])

            self.video_ids.append(video_id)
            self.qids.append(data['qid'])
            self.questions.append(data['question'])
            self.time_references.append(eval(data['hardneg_videos']))

            self.data_type.append('nextqa')

 
        qaego4d_data = json.load(open(qaego4d_anno_path, 'r'))
        for data in qaego4d_data:
            self.video_ids.append(data['video_id'])
            self.qids.append(data['sample_id'])
            self.questions.append(data['question'])

            moment_start_frame = data['moment_start_frame']
            moment_end_frame = data['moment_end_frame']
            self.time_references.append([moment_start_frame, moment_end_frame])

            self.data_type.append('qaego4d')


    def __len__(self):
        """returns the length of dataframe"""
        return len(self.data_type)


    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""
        video_id = str(self.video_ids[index])
        qid = str(self.qids[index])
        question = str(self.questions[index])
        time_reference = self.time_references[index]
        data_type = self.data_type[index]

        if data_type == 'nextqa':
            video_path = os.path.join(self.nextqa_video_path, f'{self.vid_map[video_id]}.mp4')
            pos_frames = load_video_inter(video_path, clip_frames=self.clip_frames)
            target_height, target_width = pos_frames.shape[1], pos_frames.shape[2]
            neg_vid_frames = self.neg_frames // len(time_reference)
            neg_frame_cnt, neg_frames = 0, None
            for i, neg_vid in enumerate(time_reference):
                neg_video_path = os.path.join(self.nextqa_video_path, f'{self.vid_map[str(neg_vid)]}.mp4')
                if i == len(time_reference) - 1:
                    cur_neg_frame_cnt = self.neg_frames - neg_frame_cnt
                else:
                    cur_neg_frame_cnt = neg_vid_frames
                cur_neg_frames = load_video_inter(
                    neg_video_path, 
                    clip_frames=cur_neg_frame_cnt,
                    target_height=target_height,
                    target_width=target_width
                )
                if neg_frames is None:
                    neg_frames = cur_neg_frames
                else:
                    neg_frames = np.concatenate((neg_frames, cur_neg_frames), axis=0)
                neg_frame_cnt += cur_neg_frame_cnt
            if random.random() >= 0.5:
                video = np.concatenate((pos_frames, neg_frames), axis=0)
                labels = torch.cat((torch.ones(self.clip_frames, dtype=torch.bool), torch.zeros(self.neg_frames, dtype=torch.bool)), dim=0)
            else:
                video = np.concatenate((neg_frames, pos_frames), axis=0)
                labels = torch.cat((torch.zeros(self.neg_frames, dtype=torch.bool), torch.ones(self.clip_frames, dtype=torch.bool)), dim=0)
        elif data_type == 'qaego4d':
            video_path = os.path.join(self.qaego4d_video_path, f'{video_id}.mp4')
            video, _, _, labels = load_video_intra(video_path, time_reference, self.clip_frames, self.neg_frames)

        outputs = self.processor(text=question, images=video, return_tensors="pt", padding='max_length', max_length=77)
        return {
            "video_ids": video_id,
            "qids": qid,
            "frames": outputs["pixel_values"],
            "questions": outputs['input_ids'][0][:77],
            'attention_mask': outputs['attention_mask'][0][:77],
            "labels": labels,
        }