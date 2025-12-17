#!/bin/bash

# Sign Language Detection Pipeline
# ------------------------------
# This script runs the complete sign language detection pipeline:
# 1. Video to frames conversion
# 2. Feature extraction
# 3. Segmentation (optional)
# 4. Prediction

set -e  # Exit on error

# Default values
USE_SEGMENTATION=true
MODEL="resnet50"
VIDEO_PATH="data/videos/2025-03-28-231023.webm"
OUTPUT_DIR="C:/Users/purvs/Downloads/sign-language-recognition/data/videos/output"
MODEL_PATH="models/gesture_transformer.pth"
LABELS_PATH="data/labels/word_to_label.pkl"
FEATURES_PATH="data/videos/output1/features.pkl"
FRAME_RATE=4

# Display help
function show_help {
    echo "Usage: $0 [options] -i <video_path> -o <output_dir>"
    echo "Options:"
    echo "  -i, --input      Path to input video file (required)"
    echo "  -o, --output     Output directory for results (required)"
    echo "  -m, --model      Path to model checkpoint (default: models/gesture_transformer.pth)"
    echo "  -l, --labels     Path to label mapping file (default: data/labels/word_to_label.pkl)"
    echo "  -f, --features   Path to pre-extracted features (if available)"
    echo "  -s, --seg        Use segmentation for preprocessing"
    echo "  -r, --rate       Frame rate for extraction (default: 4 frames/second)"
    echo "  -h, --help       Show this help message"
    exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -i|--input)
            VIDEO_PATH="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_PATH="$2"
            shift 2
            ;;
        -l|--labels)
            LABELS_PATH="$2"
            shift 2
            ;;
        -f|--features)
            FEATURES_PATH="$2"
            shift 2
            ;;
        -s|--seg)
            USE_SEGMENTATION=true
            shift
            ;;
        -r|--rate)
            FRAME_RATE="$2"
            shift 2
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check required arguments
if [ -z "$VIDEO_PATH" ] || [ -z "$OUTPUT_DIR" ]; then
    echo "Error: Input video and output directory are required"
    show_help
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"
FRAMES_DIR="$OUTPUT_DIR/frames"
FEATURES_FILE="$OUTPUT_DIR/features.pkl"
LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$FRAMES_DIR" "$LOG_DIR"

echo "Starting sign language detection pipeline..."
echo "Input video: $VIDEO_PATH"
echo "Output directory: $OUTPUT_DIR"

# Step 1: Convert video to frames
echo "Step 1: Converting video to frames..."
python -m src.preprocessing.video_to_frames -i "$VIDEO_PATH" -o "$FRAMES_DIR" -f "$FRAME_RATE" 2>&1 | tee "$LOG_DIR/video_to_frames.log"


    # Run segmentation script

# Step 2: Apply segmentation if requested
if [ "$USE_SEGMENTATION" = true ]; then
    cd C:/Users/purvs/Downloads/sapiens/lite/scripts/demo/torchscript
    ./seg.sh -i "$FRAMES_DIR" -o "$FRAMES_DIR/sap"
    cd C:/Users/purvs/Downloads/sign-language-recognition
    echo "Step 2: Applying segmentation..."
    python -m src.segmentation.segment -i "$FRAMES_DIR" -l "$LOG_DIR/segmentation.log" 2>&1 | tee -a "$LOG_DIR/segmentation.log"
    # Use masked images for feature extraction
    
fi
FRAMES_DIR="$FRAMES_DIR/mask"
# Step 3: Extract features or use provided features
if [ -z "$FEATURES_PATH" ]; then
    echo "Step 3: Extracting features..."
    python -m src.preprocessing.feature_extraction -i "$FRAMES_DIR" -o "$FEATURES_FILE" -m "$MODEL" -l "$LOG_DIR/feature_extraction.log" 2>&1 | tee "$LOG_DIR/feature_extraction.log"
else
    echo "Step 3: Using provided features from $FEATURES_PATH"
    FEATURES_FILE="$FEATURES_PATH"
fi

# Step 4: Prediction
echo "Step 4: Running prediction..."
# python -m src.inference.predict --model "$MODEL_PATH" --labels "$LABELS_PATH" --features "$FEATURES_FILE" --output "$OUTPUT_DIR/prediction.txt" 2>&1 | tee "$LOG_DIR/prediction.log"
python -m src.inference.predict -i "$FRAMES_DIR"
