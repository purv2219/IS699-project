@echo off
setlocal enabledelayedexpansion

REM ------------------------------
REM Sign Language Detection Pipeline (Windows Version)
REM ------------------------------

REM ==== Default values ====
set USE_SEGMENTATION=true
set MODEL=resnet50
set VIDEO_PATH=data/videos/2025-03-28-231023.webm
set OUTPUT_DIR=C:\Users\purvs\Downloads\sign-language-recognition\data\videos\output
set MODEL_PATH=models\gesture_transformer.pth
set LABELS_PATH=data\labels\word_to_label.pkl
set FEATURES_PATH=
set FRAME_RATE=4

REM ==== Parse command-line arguments ====
:parse
if "%~1"=="" goto after_parse

if "%~1"=="-i" (
    set VIDEO_PATH=%~2
    shift & shift & goto parse
)
if "%~1"=="--input" (
    set VIDEO_PATH=%~2
    shift & shift & goto parse
)
if "%~1"=="-o" (
    set OUTPUT_DIR=%~2
    shift & shift & goto parse
)
if "%~1"=="--output" (
    set OUTPUT_DIR=%~2
    shift & shift & goto parse
)

shift
goto parse

:after_parse

REM ==== Ensure required arguments exist ====
if "%VIDEO_PATH%"=="" (
    echo ERROR: Input video is required
    exit /b
)
if "%OUTPUT_DIR%"=="" (
    echo ERROR: Output directory is required
    exit /b
)

REM ==== Create directories ====
mkdir "%OUTPUT_DIR%" 2>nul
set FRAMES_DIR=%OUTPUT_DIR%\frames
set FEATURES_FILE=%OUTPUT_DIR%\features.pkl
set LOG_DIR=%OUTPUT_DIR%\logs
mkdir "%FRAMES_DIR%" 2>nul
mkdir "%LOG_DIR%" 2>nul

echo ------------------------------
echo STARTING PIPELINE
echo Video: %VIDEO_PATH%
echo Output: %OUTPUT_DIR%
echo ------------------------------

REM ==== Step 1: Convert video to frames ====
echo Step 1: Converting video to frames...
python -m src.preprocessing.video_to_frames -i "%VIDEO_PATH%" -o "%FRAMES_DIR%" -f %FRAME_RATE%

REM ==== Step 2: Segmentation (optional) ====
if "%USE_SEGMENTATION%"=="true" (
    echo Step 2: Running segmentation...
    
    pushd C:\Users\purvs\Downloads\sapiens\lite\scripts\demo\torchscript
    bash seg.sh -i "%FRAMES_DIR%" -o "%FRAMES_DIR%\sap"
    popd

    python -m src.segmentation.segment -i "%FRAMES_DIR%" -l "%LOG_DIR%\segmentation.log"
)

set FRAMES_DIR=%FRAMES_DIR%\mask

REM ==== Step 3: Feature extraction ====
if "%FEATURES_PATH%"=="" (
    echo Step 3: Extracting features...
    python -m src.preprocessing.feature_extraction -i "%FRAMES_DIR%" -o "%FEATURES_FILE%" -m "%MODEL%"
) else (
    echo Step 3: Using existing features %FEATURES_PATH%
    set FEATURES_FILE=%FEATURES_PATH%
)

REM ==== Step 4: Prediction ====
echo Step 4: Running prediction...
python -m src.inference.predict -i "%FRAMES_DIR%"

echo PIPELINE COMPLETED.
pause

