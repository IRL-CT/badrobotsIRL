#import ffmpeg and go video by video, turn into 30 fps and save as frames


import sys
sys.path.insert(0, '/home/mb2554/.local/lib/python3.8/site-packages')
import os
import ffmpeg

video_path = '../../data/videos'
video_files = os.listdir(video_path)
video_files = [f for f in video_files if f.endswith('.mp4')]

for video_file in video_files:
    # Get video name
    video_name = video_file.split('.')[0]
    print(f'Processing {video_name}...')

    # Create directory for frames
    frame_dir = f'../../data/frames/{video_name}'
    os.makedirs(frame_dir, exist_ok=True)

    # Extract frames
    ffmpeg.input(f'{video_path}/{video_file}').output(f'{frame_dir}/%d.jpg', r=30).run(overwrite_output=True)