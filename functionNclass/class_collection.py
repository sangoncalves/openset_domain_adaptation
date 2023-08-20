import random
import re
import os
from os import listdir
from os.path import join
from random import Random
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image, ImageFilter
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import RandomHorizontalFlip, RandomCrop, RandomResizedCrop, ColorJitter
import tempfile
from collections import defaultdict
from torch.utils.data import Subset

def apply_data_augmentation(img):
    # Define the data augmentation transformations
    augmentations = [
        RandomHorizontalFlip(p=0.5),
        RandomCrop(size=(224, 224), padding=4),
        RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3.0/4.0, 4.0/3.0)),
        ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
    ]

    # Apply the transformations to the input image
    for transform in augmentations:
        img = transform(img)

    return img

def get_frames_by_class(txt_file_path, n_frames):
  p = txt_file_path

  video_label = []
  classes = {}
  # Open the text file for reading
  with open(p, 'r') as file:
      # Iterate over each line in the file
      for line in file:
          # Strip any newline characters from the line
          line_split = line.strip().split()
          # Append the line to the self.video_label and classes list
          classes[line_split[0]]=line_split[-1]
          video_label.append(tuple(line_split[1:]))
  # create the inverse mapping
  class_id_to_name = {v: k for k, v in classes.items()}

  return video_label, classes, class_id_to_name

class VideoDatasetSourceAndTarget:
    def __init__(self, source_dataset, target_dataset):
        self.source_dataset = source_dataset
        self.target_dataset = target_dataset
        self.output_dim = (source_dataset.output_dim, target_dataset.output_dim)
        self.unknown_label=None
        # Get the size of the tensors for the first sample in the source and target datasets
        source_size = len(source_dataset[0][0]), *source_dataset[0][0].size()[1:]
        target_size = len(target_dataset[0][0]), *target_dataset[0][0].size()[1:]

        # Store the size of the tensors as instance variables
        self.source_size = source_size
        self.target_size = target_size

    def __len__(self):
        return max([len(self.source_dataset), len(self.target_dataset)])

    def __getitem__(self, index):
        source_index = index % len(self.source_dataset)
        source_data = self.source_dataset[source_index]

        target_index = index % len(self.target_dataset)
        target_data = self.target_dataset[target_index]
        return (source_index, *source_data, target_index, *target_data)

class VideoDataset(Dataset):

    def __init__(self, dataset_path, txt_file_path, n_frames=16, n_clips=4, frame_size=224, normalize=True, train=True, augmentation=False, fake_label=False):
        super().__init__()
        self.dataset_root = dataset_path
        self.n_frames = n_frames
        self.n_clips = n_clips
        self.normalize = normalize
        self.mean, self.std = None, None
        self.reorder_shape = False
        self.video_label, self.classes, self.class_id_to_name = get_frames_by_class(txt_file_path, n_frames)
        self.output_dim = len(self.classes) if not fake_label else max(self.classes.values()) + 1

        self.train = train
        self.augmentation = augmentation

        # fix size
        if isinstance(frame_size, int):
            self.frame_size = (frame_size, frame_size)
        else:
            self.frame_size = frame_size
        
    def calculate_new_labels(self):
        unique_old_labels = set(label for _, label in self.video_label)
        new_labels = {old_label: i for i, old_label in enumerate(unique_old_labels)}
        self.new_labels = new_labels

    def __len__(self):
        return len(self.video_label)

    def load_frame(self, path):
        return Image.open(path)

    def gen_transformation_pipeline(self):
        if (self.frame_size[0] == 224):
            s = (256, 256)
        elif (self.frame_size[0] == 112):
            s = (128, 128)
        else:
            raise Exception("Size is not supported")
        transformations = [(TF.resize, s), (TF.to_tensor,)]
        if self.normalize and self.mean is not None:
            transformations.append((TF.normalize, self.mean, self.std))
        return transformations

    def transform_frame(self, frame, transformations):
        for transform, *args in transformations:
            frame = transform(frame, *args)
        return frame

    def frame2video_tensor(self, reduced_frames):
        tensors = torch.stack([self.transform_frame(frame, self.gen_transformation_pipeline())
                               for frame in reduced_frames])
        # tensors = tensors.reshape(self.n_clips, self.n_frames, *tensors.size()[1:])
        if self.reorder_shape:
            tensors = tensors.permute(0, 2, 1, 3, 4)
        return tensors

    def video2frames(self, video_path):
        img_ext = ['jpg', 'png', 'jpeg', 'gif', 'tiff', 'psd', 'eps']
        frames_path = os.listdir(video_path)
        frames = [join(video_path, f) for f in frames_path if f.split('.')[-1].lower() in img_ext]
        return sorted(frames)

    # def frames2indices(self, num_frames, n_clips):
    #     num_frames_clip = num_frames // n_clips
    #     tick = num_frames_clip / self.n_frames
    #     # pick the central frame in each segment
    #     indexes = np.array([int(tick / 2.0 + tick * x) for x in range(self.n_frames)])  
    #     indexes = np.tile(indexes, n_clips)
    #     for i in range(n_clips):
    #         indexes[i * self.n_frames : (i + 1) * self.n_frames] += num_frames_clip * i
    #     return indexes

    def frames2indices(self, num_frames):
      indices = np.linspace(0, num_frames - 1, num=self.n_frames, dtype=int)
      return indices


    def __getitem__(self, index):
        video_path, label = self.video_label[index]
        frame_paths = self.video2frames(video_path)
        n_frames = len(frame_paths)
        indices = self.frames2indices(n_frames) #choose "relevant" indexes. Closer frames are too similar and can be discarded. #uniform dist according the frames
        
        reduced_frames = []
        for i in indices:
            frame = self.load_frame(frame_paths[i])
            if self.train and self.augmentation: # Apply augmentation only if train=True and augmentation=True
                frame = apply_data_augmentation(frame)
            reduced_frames.append(frame)
        
        tensor = self.frame2video_tensor(reduced_frames)     
        return tensor, int(label) # return tensor

class ExtendedSubset(Subset):
    def __init__(self, dataset, indices, classes, video_label, class_id_to_name):
        super().__init__(dataset, indices)
        self.class_id_to_name = class_id_to_name
        # self.classes = {k: classes[k] for k in self.indices}
        self.classes = {idx: self.class_id_to_name[video_label[idx][1]] for idx in self.indices}
        self.classes = classes
        self.video_label = [video_label[i] for i in self.indices]
        self.index_mapping = {original_idx: subset_idx for subset_idx, original_idx in enumerate(self.indices)}

class ClassObservationsSamplerVideoDatasetSourceAndTarget:
    def __init__(self, video_dataset, num_observations):
        self.video_dataset = video_dataset
        self.num_observations = num_observations

        # Get subset indices for source and target datasets
        self.source_indices = self._get_indices_by_class(self.video_dataset.source_dataset)
        self.target_indices = self._get_indices_by_class(self.video_dataset.target_dataset)

        # Create a subset of the source and target datasets
        self.source_dataset = ExtendedSubset(
            self.video_dataset.source_dataset, 
            self.source_indices, 
            self.video_dataset.source_dataset.classes,
            self.video_dataset.source_dataset.video_label,
            self.video_dataset.source_dataset.class_id_to_name
        )
        self.target_dataset = ExtendedSubset(
            self.video_dataset.target_dataset, 
            self.target_indices, 
            self.video_dataset.target_dataset.classes,
            self.video_dataset.target_dataset.video_label,
            self.video_dataset.target_dataset.class_id_to_name

        )

        self.output_dim = self.video_dataset.output_dim

        # Get the size of the tensors for the first sample in the source and target datasets
        source_size = len(self.source_dataset[0][0]), *self.source_dataset[0][0].size()[1:]
        target_size = len(self.target_dataset[0][0]), *self.target_dataset[0][0].size()[1:]

        # Store the size of the tensors as instance variables
        self.source_size = source_size
        self.target_size = target_size

    def _get_indices_by_class(self, dataset):
        class_indices = defaultdict(list)
        for idx in range(len(dataset.video_label)):
            _, class_id = dataset.video_label[idx]
            class_indices[int(class_id)].append(idx)
        indices = []
        for class_id, class_idx in class_indices.items():
            np.random.shuffle(class_idx)
            indices.extend(class_idx[:self.num_observations])
        return indices

    def __len__(self):
        return max(len(self.source_dataset), len(self.target_dataset))

    def __getitem__(self, index):
      # Get source and target indices based on subset indices
      source_index = self.source_indices[index % len(self.source_indices)]
      source_subset_index = self.source_dataset.index_mapping[source_index]
      source_data, source_label = self.source_dataset[source_subset_index]

      target_index = self.target_indices[index % len(self.target_indices)]
      target_subset_index = self.target_dataset.index_mapping[target_index]
      target_data, target_label = self.target_dataset[target_subset_index]

      return source_index, source_data, source_label, target_index, target_data, target_label

class ClassObservationsSamplerVideoDatasetTarget:
    def __init__(self, video_dataset, num_observations):
        self.video_dataset = video_dataset
        self.num_observations = num_observations

        # Get subset indices for source and target datasets
        self.source_indices = self._get_indices_by_class(self.video_dataset)

        # Create a subset of the source and target datasets
        self.source_dataset = ExtendedSubset(
            self.video_dataset, 
            self.source_indices, 
            self.video_dataset.classes,
            self.video_dataset.video_label,
            self.video_dataset.class_id_to_name
        )

        self.output_dim = self.video_dataset.output_dim

        # Get the size of the tensors for the first sample in the source and target datasets
        source_size = len(self.source_dataset[0][0]), *self.source_dataset[0][0].size()[1:]

        # Store the size of the tensors as instance variables
        self.source_size = source_size

        # Assign video_label and classes from source_dataset
        self.video_label = self.source_dataset.video_label
        self.classes = self.source_dataset.classes
        self.class_id_to_name = self.source_dataset.class_id_to_name  # Carry over class_id_to_name attribute

    def _get_indices_by_class(self, dataset):
        class_indices = defaultdict(list)
        for idx in range(len(dataset.video_label)):
            _, class_id = dataset.video_label[idx]
            class_indices[int(class_id)].append(idx)
        indices = []
        for class_id, class_idx in class_indices.items():
            np.random.shuffle(class_idx)
            indices.extend(class_idx[:self.num_observations])
        return indices

    def __len__(self):
      return len(self.source_dataset)


    def __getitem__(self, index):
      # Get source and target indices based on subset indices
      source_index = self.source_indices[index % len(self.source_indices)]
      source_subset_index = self.source_dataset.index_mapping[source_index]
      source_data = self.source_dataset[source_subset_index]

      return (source_index, *source_data)        