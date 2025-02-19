import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import Dataset
import numpy as np
import h5py
import os
import cv2
from torchvision.models import ViT_B_16_Weights
import torch.nn as nn


class VideoFrameProcessor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # Initialize ViT model and transformation
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Load pre-trained ViT model
        self.vit_model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(self.device)
        self.feature_extractor = nn.Sequential(*list(self.vit_model.children())[:-1])
        self.encoder = self.feature_extractor[1]

        self.vit_model.eval()  # Set to evaluation mode

    def process_frames(self, frames, output_path):
        """
        Process video frames through ViT and store in efficient HDF5 format
        
        Args:
            frames (np.ndarray): Input frames (N, H, W, C)
            output_path (str): Path to save processed features
        
        Returns:
            dict: Metadata about processed frames
        """
        processed_features = []
        
        # Process each frame
        with torch.no_grad():
            for frame in frames:
                # Transform frame
                transformed_frame = self.transform(frame).unsqueeze(0).to(self.device)
                #print(transformed_frame.shape)
                # Extract features
                transformed_frame = self.vit_model._process_input(transformed_frame)
                n = transformed_frame.size(0)
                batch_class_tokens = self.vit_model.class_token.expand(n, -1, -1)
                transformed_frame = torch.cat((batch_class_tokens, transformed_frame), dim=1)
                features = self.encoder(transformed_frame)
                features = features[:, 0]
                #features = self.vit_model.encoder(transformed_frame)
                #features = self.forward_features(transformed_frame)
                processed_features.append(features.cpu().numpy())

        print(f'Processed {len(frames)} frames.')
        # Convert to numpy array
        processed_features = np.array(processed_features)
        
        # Save to HDF5 for efficient storage
        with h5py.File(output_path, 'w') as hf:
            hf.create_dataset('features', data=processed_features, compression='gzip')
        
        return {
            'num_frames': len(frames),
            'feature_shape': processed_features.shape,
            'output_path': output_path
        }

class VideoFeatureDataset(torch.utils.data.Dataset):
    def __init__(self, h5_path, labels):
        """
        Dataset for efficient loading of processed video features
        
        Args:
            h5_path (str): Path to HDF5 file with features
            labels (list): Corresponding labels for features
        """
        self.features = h5py.File(h5_path, 'r')['features']
        self.labels = torch.tensor(labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), self.labels[idx]

# Example usage
if __name__ == '__main__':
    
    #get videos and video frames from the folder
    video_path = '../../data/frames'
    video_files = os.listdir(video_path)

    # Process each video
    processor = VideoFrameProcessor()


    for frames_folder in video_files:
        #check if there is an h5 file for this video, if so skip
        if os.path.exists(f'../../data/features/{frames_folder}.h5'):
            print(f'{frames_folder} already processed.')
            continue
        print(f'Processing {frames_folder}...')
        #load all frames to a np arrat
        frames = []
        frames_files = os.listdir(f'{video_path}/{frames_folder}')
        for frame_file in frames_files:
            frame = cv2.imread(f'{video_path}/{frames_folder}/{frame_file}')
            frames.append(frame)
            #stop at frame 100
            #if len(frames) == 100:
            #    break
            
        print(f'Read {frames_folder}.')
        #make the frames into a np array
        frames = np.array(frames)
        print(frames.shape)
        metadata = processor.process_frames(frames, f'../../data/features/{frames_folder}.h5')
    
    #frames = np.random.randint(0, 256, (10, 480, 640, 3), dtype=np.uint8)
    #labels = [0, 1, 0, 1, 1, 0, 0, 1, 1, 0]  # Example labels
    
    #processor = VideoFrameProcessor()
    #metadata = processor.process_frames(frames, '../../data/features/video_features.h5')
    
    # Create dataset for training
    #dataset = VideoFeatureDataset('../../data/features/video_features.h5', labels)
    
    print(metadata)