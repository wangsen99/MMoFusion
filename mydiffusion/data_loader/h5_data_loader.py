import torch
import h5py
import numpy as np
from torch.utils.data import DataLoader
import pdb as db
import os

class AllSpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, style_dim, sequence_length, npy_root="../../process", 
                 version='v0', dataset='BEAT'):
        self.h5 = h5py.File(h5file, "r")
        self.len = len(self.h5.keys())
        self.motion_dim = motion_dim
        self.style_dim = style_dim
        self.version = version
        
        gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))

        self.id = [[int(self.h5[str(i)]["speaker_id"][:][0])-1] for i in range(len(self.h5.keys()))]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        self.emotion = [self.h5[str(i)]["emotion"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        self.h5.close()
        self.sequence_length = sequence_length
        # if "v0" in self.version:
        #     self.gesture_vel = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture]
        #     self.gesture_acc = [np.concatenate((np.zeros([1, self.motion_dim]), i[1:] - i[:-1]), axis=0) for i in self.gesture_vel]
        print("Total clips:", len(self.gesture), 'length:', sequence_length)
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):
        total_frame_len = self.audio[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        audio = self.audio[idx][start_frame:end_frame]
        text = self.text[idx][start_frame:end_frame]
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        posrat = self.gesture[idx][start_frame:end_frame]
        gesture = posrat
        
        gesture = torch.FloatTensor(gesture)
        speaker = np.zeros([self.style_dim])
        speaker[self.id[idx]] = 1
        speaker = torch.FloatTensor(speaker)

        emotion = self.emotion[idx][start_frame:end_frame]
        emotion = torch.from_numpy(emotion).int()

        return textaudio, gesture, speaker, emotion
        
class UpperSpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, style_dim, sequence_length, npy_root="../../process", 
                 version='v0', dataset='BEAT'):
        self.h5 = h5py.File(h5file, "r")
        self.len = len(self.h5.keys())
        self.motion_dim = motion_dim
        self.style_dim = style_dim
        self.version = version
        
        gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))

        self.id = [[int(self.h5[str(i)]["speaker_id"][:][0])-1] for i in range(len(self.h5.keys()))]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        self.emotion = [self.h5[str(i)]["emotion"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        self.h5.close()
        self.sequence_length = sequence_length
        print("Total upper body clips:", len(self.gesture), 'length:', sequence_length)
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):
        total_frame_len = self.audio[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        audio = self.audio[idx][start_frame:end_frame]
        text = self.text[idx][start_frame:end_frame]
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        posrat = self.gesture[idx][start_frame:end_frame]
        gesture = posrat
        
        gesture = torch.FloatTensor(gesture)

        upper_gesture = gesture[:, 6*3:192*3]

        speaker = np.array(self.id[idx])
        speaker = torch.from_numpy(speaker).int()

        emotion = self.emotion[idx][start_frame:end_frame]
        emotion = torch.from_numpy(emotion).int()

        return textaudio, upper_gesture, speaker, emotion
    
class WholeSpeechGestureDataset(torch.utils.data.Dataset):
    def __init__(self, h5file, motion_dim, style_dim, sequence_length, npy_root="../../process", 
                 version='v0', dataset='BEAT'):
        self.h5 = h5py.File(h5file, "r")
        self.len = len(self.h5.keys())
        self.motion_dim = motion_dim
        self.style_dim = style_dim
        self.version = version
        
        gesture_mean = np.load(os.path.join(npy_root, "gesture_" + dataset + "_mean_" + self.version + ".npy"))
        gesture_std = np.load(os.path.join(npy_root, "gesture_" + dataset + "_std_" + self.version + ".npy"))

        self.id = [[int(self.h5[str(i)]["speaker_id"][:][0])-1] for i in range(len(self.h5.keys()))]
        self.audio = [self.h5[str(i)]["audio"][:] for i in range(len(self.h5.keys()))]
        self.text = [self.h5[str(i)]["text"][:] for i in range(len(self.h5.keys()))]
        self.emotion = [self.h5[str(i)]["emotion"][:] for i in range(len(self.h5.keys()))]
        self.gesture = [(self.h5[str(i)]["gesture"][:] - gesture_mean) / gesture_std for i in range(len(self.h5.keys()))]
        self.h5.close()
        self.sequence_length = sequence_length
        print("Total upper body clips:", len(self.gesture), 'length:', sequence_length)
        self.segment_length = sequence_length

    def __len__(self):
        return len(self.gesture)

    def __getitem__(self, idx):
        total_frame_len = self.audio[idx].shape[0]
        start_frame = np.random.randint(0, total_frame_len - self.segment_length)
        end_frame = start_frame + self.segment_length
        audio = self.audio[idx][start_frame:end_frame]
        text = self.text[idx][start_frame:end_frame]
        textaudio = np.concatenate((audio, text), axis=-1)
        textaudio = torch.FloatTensor(textaudio)
        posrat = self.gesture[idx][start_frame:end_frame]
        gesture = posrat
        
        gesture = torch.FloatTensor(gesture)

        # upper_gesture = gesture[:, 6*3:192*3]

        speaker = np.array(self.id[idx])
        speaker = torch.from_numpy(speaker).int()

        emotion = self.emotion[idx][start_frame:end_frame]
        emotion = torch.from_numpy(emotion).int()

        return textaudio, gesture, speaker, emotion

class RandomSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        while True:
            yield np.random.randint(self.min_id, self.max_id)


class SequentialSampler(torch.utils.data.Sampler):
    def __init__(self, min_id, max_id):
        self.min_id = min_id
        self.max_id = max_id
    def __iter__(self):
        return iter(range(self.min_id, self.max_id))