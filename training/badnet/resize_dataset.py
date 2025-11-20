#!/usr/bin/env python3
"""Convert images to preprocessed numpy arrays."""

import os
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import argparse

def convert_to_npy(input_dir, output_dir, size=(224, 224)):
    """Convert all images to preprocessed numpy arrays."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Transform: resize, convert to tensor format, normalize
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    participants = [d for d in os.listdir(input_dir) 
                   if os.path.isdir(os.path.join(input_dir, d))]
    
    total_converted = 0
    skipped_participants = 0
    
    for participant in participants:
        input_participant_dir = os.path.join(input_dir, participant)
        output_participant_dir = os.path.join(output_dir, participant)
        
        # Check if participant already processed
        if os.path.exists(output_participant_dir):
            # Count JPG files in input and NPY files in output
            input_images = [f for f in os.listdir(input_participant_dir) if f.endswith('.jpg')]
            existing_npy = [f for f in os.listdir(output_participant_dir) if f.endswith('.npy')]
            
            # Skip if all images already converted
            if len(existing_npy) >= len(input_images):
                print(f"Skipping {participant}: already converted ({len(existing_npy)} NPY files)")
                skipped_participants += 1
                continue
            else:
                print(f"Resuming {participant}: {len(existing_npy)}/{len(input_images)} already done")
        
        os.makedirs(output_participant_dir, exist_ok=True)
        
        images = [f for f in os.listdir(input_participant_dir) if f.endswith('.jpg')]
        
        print(f"Processing {participant}: {len(images)} images")
        
        for img_name in tqdm(images, desc=participant):
            input_path = os.path.join(input_participant_dir, img_name)
            output_path = os.path.join(output_participant_dir, img_name.replace('.jpg', '.npy'))
            
            # Skip if already exists
            if os.path.exists(output_path):
                continue
            
            try:
                # Load and transform image
                img = Image.open(input_path).convert('RGB')
                img_tensor = transform(img)
                
                # Convert to numpy and save
                img_array = img_tensor.numpy()  # Shape: (3, 224, 224)
                np.save(output_path, img_array)
                total_converted += 1
                
            except Exception as e:
                print(f"Error processing {input_path}: {e}")
    
    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"Converted: {total_converted} new images")
    print(f"Skipped: {skipped_participants} participants (already done)")
    print(f"{'='*60}")
    
    # Print size comparison
    if total_converted > 0:
        # Find first participant with NPY files
        for participant in participants:
            output_participant_dir = os.path.join(output_dir, participant)
            if os.path.exists(output_participant_dir):
                npy_files = [f for f in os.listdir(output_participant_dir) if f.endswith('.npy')]
                if npy_files:
                    sample_npy = os.path.join(output_participant_dir, npy_files[0])
                    size_mb = os.path.getsize(sample_npy) / 1024 / 1024
                    print(f"Sample NPY file size: {size_mb:.2f} MB")
                    
                    # Estimate total size
                    total_npy = sum(len([f for f in os.listdir(os.path.join(output_dir, p)) 
                                        if f.endswith('.npy')])
                                   for p in os.listdir(output_dir)
                                   if os.path.isdir(os.path.join(output_dir, p)))
                    print(f"Total NPY files: {total_npy}")
                    print(f"Estimated total size: {size_mb * total_npy:.2f} MB ({size_mb * total_npy / 1024:.2f} GB)")
                    break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="../../../data/frames")
    parser.add_argument("--output_dir", type=str, default="../../../data/frames_npy")
    parser.add_argument("--size", type=int, default=224)
    args = parser.parse_args()
    
    convert_to_npy(args.input_dir, args.output_dir, (args.size, args.size))