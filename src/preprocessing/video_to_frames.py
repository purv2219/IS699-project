#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video to Frames Conversion Utility
---------------------------------
This module extracts frames from video files at a specified frame rate.
"""

import cv2
import os
import argparse

def video_to_images(video_path, output_folder, frame_rate=4):
    """
    Convert a video file to a sequence of image frames.
    
    Args:
        video_path (str): Path to the input video file
        output_folder (str): Directory where the frames will be saved
        frame_rate (int): Number of frames to extract per second
        
    Returns:
        int: Number of frames extracted
    """
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        return 0
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    frame_count = 0
    image_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            image_path = os.path.join(output_folder, f"frame_{image_count:05d}.jpg")
            cv2.imwrite(image_path, frame)
            image_count += 1
        frame_count += 1
    
    cap.release()
    print(f"Saved {image_count} frames in {output_folder}")
    return image_count

def main():
    """Command line interface for video to frames conversion."""
    parser = argparse.ArgumentParser(description="Convert video to images")
    parser.add_argument("-i", "--input", required=True, help="Path to the input video file")
    parser.add_argument("-o", "--output", required=True, help="Path to the output folder")
    parser.add_argument("-f", "--frame_rate", type=int, default=4, 
                        help="Frames to extract per second (default: 4)")
    args = parser.parse_args()
    
    video_to_images(args.input, args.output, args.frame_rate)

if __name__ == "__main__":
    main() 