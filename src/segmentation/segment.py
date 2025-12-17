#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Image Segmentation Utilities
---------------------------
Utilities for processing segmentation masks and applying them to images.
"""

import numpy as np
from PIL import Image
import os
import cv2
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_directories(base_folder):
    """
    Setup input and output directories for segmentation process.
    
    Args:
        base_folder (str): Base directory for input/output
        
    Returns:
        tuple: (seg_folder, output_folder) paths
    """
    seg_folder = os.path.join(base_folder, 'sap')  # Segmentation folder
    output_folder = os.path.join(base_folder, 'mask')  # Output folder for masks
    
    if not os.path.exists(seg_folder):
        raise FileNotFoundError(f"Segmentation folder {seg_folder} does not exist.")
    
    os.makedirs(output_folder, exist_ok=True)
    
    return seg_folder, output_folder

def apply_segmentation_mask(seg_path, img_path, output_folder, target_size=(224, 224)):
    """
    Apply segmentation mask to an image.
    
    Args:
        seg_path (str): Path to segmentation numpy file
        img_path (str): Path to the original image
        output_folder (str): Directory to save output files
        target_size (tuple): Size to resize images to (height, width)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Load segmentation mask
        seg_mask = np.load(seg_path)
        
        # Load the corresponding image
        if not os.path.exists(img_path):
            logger.warning(f"Image {img_path} not found. Skipping...")
            return False
            
        img = Image.open(img_path).convert('RGB')
        img_array = np.array(img, dtype='uint8')
        
        # Resize segmentation and image
        img_resized = cv2.resize(img_array, target_size, interpolation=cv2.INTER_NEAREST)
        seg_resized = cv2.resize(seg_mask, target_size, interpolation=cv2.INTER_NEAREST)
        
        # Create binary mask (class 5 or 14)
        mask1 = (seg_resized == 14).astype(np.uint8)  # class 14
        mask2 = (seg_resized == 5).astype(np.uint8)   # class 5
        combined_mask = mask1 + mask2
        
        # Expand mask dimensions and tile for RGB
        mask_expanded = np.expand_dims(combined_mask, axis=-1)
        mask_tiled = np.tile(mask_expanded, (1, 1, 3))
        
        # Apply mask to image
        masked_img = np.multiply(mask_tiled, img_resized)
        
        # Extract base filename
        base_filename = os.path.basename(seg_path).replace('_seg.npy', '')
        
        # Save masked output as .npy
        npy_output_path = os.path.join(output_folder, f"{base_filename}_mask.npy")
        np.save(npy_output_path, masked_img)
        
        # Save masked output as image
        img_output = Image.fromarray(masked_img)
        png_output_path = os.path.join(output_folder, f"{base_filename}_mask.jpg")
        img_output.save(png_output_path)
        
        logger.info(f"Processed: {png_output_path}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {seg_path}: {str(e)}")
        return False

def process_segmentation_folder(base_folder):
    """
    Process all segmentation files in a folder.
    
    Args:
        base_folder (str): Base directory for input/output
        
    Returns:
        int: Number of successfully processed files
    """
    try:
        seg_folder, output_folder = setup_directories(base_folder)
        
        success_count = 0
        total_count = 0
        
        # Process each segmentation file
        for filename in os.listdir(seg_folder):
            if filename.endswith('_seg.npy'):
                total_count += 1
                seg_path = os.path.join(seg_folder, filename)
                
                # Derive the image path from segmentation filename
                img_filename = filename.replace('_seg.npy', '.png')
                img_path = os.path.join(base_folder, img_filename)
                
                if apply_segmentation_mask(seg_path, img_path, output_folder):
                    success_count += 1
                img_filename = filename.replace('_seg.npy', '.jpg')
                img_path = os.path.join(base_folder, img_filename)
                
                if apply_segmentation_mask(seg_path, img_path, output_folder):
                    success_count += 1
        
        logger.info(f"Segmentation complete: {success_count}/{total_count} files processed successfully")
        return success_count
        
    except Exception as e:
        logger.error(f"Error in segmentation processing: {str(e)}")
        return 0

def main():
    """Command line interface for segmentation processing."""
    parser = argparse.ArgumentParser(description='Process segmentation masks and images.')
    parser.add_argument('-i', '--input', required=True, help='Input directory containing images and segmentation data')
    parser.add_argument('-l', '--log', help='Path to log file')
    args = parser.parse_args()
    
    # Configure file logging if specified
    if args.log:
        file_handler = logging.FileHandler(args.log)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
    
    process_segmentation_folder(args.input)

if __name__ == "__main__":
    main() 