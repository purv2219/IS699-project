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
OUTPUT_DIR="data/videos/output1"
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

    # Run segmentation script

# Step 2: Apply segmentation if requested
#!/bin/bash

# Define the base directory
BASE_DIR="C:\Users\purvs\Downloads\png-segments"

# Loop through folders from 001 to 201
for i in $(seq -w 1 201); do
    FRAMES_DIR="$BASE_DIR/$i"

    if [ -d "$FRAMES_DIR" ]; then
        echo "Processing folder: $FRAMES_DIR"

        if [ "$USE_SEGMENTATION" = false ]; then
            cd C:/Users/purvs/Downloads/sapiens/lite/scripts/demo/torchscript
            ./seg.sh -i "$FRAMES_DIR" -o "$FRAMES_DIR/sap"
            
            cd C:/Users/purvs/Downloads/sign-language-recognition
            echo "Step 2: Applying segmentation..."
            python -m src.segmentation.segment -i "$FRAMES_DIR" -l "$LOG_DIR/segmentation.log" 2>&1 | tee -a "$LOG_DIR/segmentation.log"

            # Use masked images for feature extraction
            # FRAMES_DIR="$FRAMES_DIR/mask"
        fi

        # Step 3: Extract features
        echo "Step 3: Extracting features..."
        
    else
        echo "Skipping folder: $FRAMES_DIR (does not exist)"
    fi
done

    # echo "Step 3: Using provided features from $FEATURES_PATH"
    # FEATURES_FILE="$FEATURES_PATH"
python -m src.preprocessing.feature_extraction -i "$BASE_DIR" -t train
python -m src.preprocessing.feature_extraction -i "$BASE_DIR" -t test

# Step 4: Prediction
echo "Step 4: Running prediction..."
# python -m src.inference.predict --model "$MODEL_PATH" --labels "$LABELS_PATH" --features "$FEATURES_FILE" --output "$OUTPUT_DIR/prediction.txt" 2>&1 | tee "$LOG_DIR/prediction.log"
python -m src.utils.lstm 
echo "Pipeline completed! Results saved to $OUTPUT_DIR/prediction.txt"
cat "$OUTPUT_DIR/prediction.txt" 