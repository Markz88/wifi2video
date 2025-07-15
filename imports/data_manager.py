"""
LOAD DATA from file
"""

import os
from os import walk
import re
import torch
import numpy as np

import torchvision.transforms as transforms

import scipy.io
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

from torch.utils.data import Dataset

from PIL import Image

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [tryint(c) for c in re.split('([0-9]+)', s)]

# From categorical labels to numerical or dummy
def categorical_numerical_dummy(labels):

    # integer encode (from categorical to numerical)
    label_encoder = LabelEncoder()
    numerical_encoded = label_encoder.fit_transform(labels)

    # binary encode (OneHotEncoder/Dummy)
    onehot_encoder = OneHotEncoder(sparse_output=False)
    reshape_encoded = numerical_encoded.reshape(len(numerical_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(reshape_encoded)

    return labels, numerical_encoded, onehot_encoded

# Load train and test videos
def load_video(opt, usecase):

    # Set the dataset root directory
    if opt['dataroot'] == '':
        # Set 'data' as default root directory if not specified
        data_path = './data/{}'.format(opt['dataset_name'])
    else:
        # Set as default root directory the one specified in CONFIG file
        data_path = opt['dataroot']+opt['dataset_name']

    if opt['videoGAN']['img_ch'] == 1:
        # Image transformations: scaling, tensor creation and normalization (Grayscale)
        transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
                                        transforms.Resize((opt['videoGAN']['img_size'],opt['videoGAN']['img_size'])),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5,), (0.5,))])
    else:
        # Image transformations: scaling, tensor creation and normalization (RGB)
        transform = transforms.Compose([transforms.Resize((opt['videoGAN']['img_size'], opt['videoGAN']['img_size'])),
                                        transforms.ToTensor(),
                                        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ])

    # Videos data collection
    videosCollection = [] # frames of processed videos
    labels = [] # classes
    filenames = [] # filenames

    # List of video classes
    classes = [c for c in os.listdir(os.path.join(data_path, usecase, opt['test']['dir_video']))
           if not c.startswith('.') and os.path.isdir(os.path.join(data_path, usecase, opt['test']['dir_video'], c))]
    
    # Sort class names in natural order
    classes = sorted(classes, key=alphanum_key)

    # For each class
    for category in classes:
        # List of video directories
        videos = [v for v in os.listdir(os.path.join(data_path, usecase, opt['test']['dir_video'], category))
          if not v.startswith('.')]

        # Sort videos by directory name in natural order
        videos = sorted(videos, key=alphanum_key)

        # For each video
        for video in videos:
            video_path = os.path.join(data_path, usecase, opt['test']['dir_video'], category, video)
    
            # Skip non-directories like .DS_Store
            if not os.path.isdir(video_path):
                continue

            # List of processed frames
            _, _, frames = next(walk(video_path), (None, None, []))
            
            # Sort frames in natural order
            frames = sorted(frames, key=alphanum_key)

            # Collect processed frame
            curr_frames = []

            # For each frame
            for frame in frames:
                # Apply image transformation
                img = transform(Image.open(os.path.join(data_path,usecase,opt['test']['dir_video'], category, video, frame)))
                # Collect processed frame
                curr_frames.append(img)

            # Collect processed video
            videosCollection += [curr_frames]
            # Collect the class of processed video
            labels.append(category)
            # Collect the filename of processed video
            filenames.append(video)

    # Flatten labels array
    labels = np.array(labels).flatten()
    # Encoding labels
    labels, numerical, dummy = categorical_numerical_dummy(labels)
    # Flatten filenames array
    filenames = np.array(filenames).astype(np.float32).flatten()


    return videosCollection, numerical, filenames

# Load train and test signals
def load_sgnls(opt, usecase):

    # Set the dataset root directory
    if opt['dataroot'] == '':
        # Set 'data' as default root directory if not specified
        data_path = opt['dataroot'] = './data/{}'.format(opt['dataset_name'])
    else:
        # Set as default root directory the one specified in CONFIG file
        data_path = opt['dataroot']+opt['dataset_name']

    # Signals data collection
    signalsCollection = []
    labels = []

    
    # List of signal classes
    classes = [c for c in os.listdir(os.path.join(data_path, usecase, opt['test']['dir_sgnl']))
           if not c.startswith('.') and os.path.isdir(os.path.join(data_path, usecase, opt['test']['dir_sgnl'], c))]

    # Sort class names in natural order
    classes = sorted(classes, key=alphanum_key)

    # For each class
    for category in classes:
        # List of signal directories
        signals = [v for v in os.listdir(os.path.join(data_path, usecase, opt['test']['dir_sgnl'], category))
          if not v.startswith('.')]

        # Sort signals by directory name in natural order
        signals = sorted(signals, key=alphanum_key)

        # For each signal
        for sgnl in signals:
            # Load signal features
            sgnl_features = scipy.io.loadmat(os.path.join(data_path,usecase,opt['test']['dir_sgnl'],category,sgnl,'Features.mat')) # load .mat file
            # Collect signal CSI extracted amplitudes
            signalsCollection.append(sgnl_features['fillMed'])
            # Collect the class of processed signal
            labels.append(category)

    # From array to tensor
    signalsCollection = torch.as_tensor(signalsCollection, dtype=torch.float32)
    # Flatten labels array
    labels = np.array(labels).flatten()
    # Encoding labels
    labels, numerical, dummy = categorical_numerical_dummy(labels) # encode categorical labels

    return signalsCollection, numerical

# Create dataset pairs (video,signal)
class load_data(Dataset):

    def __init__(self, opt, usecase):
        # Load processed training videos, labels, filename
        self.videos, self.labels, self.filenames = load_video(opt, usecase)
        # Load processed training signals
        self.signals, _ = load_sgnls(opt, usecase)

    def __len__(self):
        # Total number of samples
        return len(self.videos)

    def __getitem__(self, index):
        # Generate each sample of data (frames, signal, label, filename)

        # Video frames
        X = self.videos[index]
        # Stack video frames in a tensor - shape [D x C x H x W] with D number of frames
        X = torch.stack(X)
        # Reshape for using in pytorch 3D models -> new shape [C x D x H x W]
        X = X.permute(1, 0, 2, 3)

        # Current signal synchronized to video
        y = self.signals[index]

        # Sample class
        z = self.labels[index]

        # Video filename
        k = self.filenames[index]

        return X, y, z, k
'''
# Main function
if __name__ == '__main__': # required for CUDA usage
    # Read CONFIG file for training and testing parameters
    with open("config.yaml", 'r') as stream:
        opt = yaml.load(stream) # CONFIG parameters

    load_imgs(opt, train=True)
'''