import argparse
import os
import subprocess
import sys

# ------------------------------
# Sign Language Detection Pipeline (Python Version)
# ------------------------------

def run_command(cmd, log_file=None):
    """
    Run a shell command and optionally log output to a file.
    """
    print(f"Running: {' '.join(cmd)}")
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    ) as process:
        output_lines = []
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)
        process.wait()

        if log_file:
            with open(log_file, "a", encoding="utf-8") as f:
                f.writelines(output_lines)

        if process.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")


def main():
    parser = argparse.ArgumentParser(description="Sign Language Detection Pipeline")

    parser.add_argument("-i", "--input", required=True, help="Path to input video file")
    parser.add_argument("-o", "--output", required=True, help="Output directory")
    parser.add_argument("-m", "--model", default="resnet50", help="Model name")
    parser.add_argument("--model_path", default="models/gesture_transformer.pth")
    parser.add_argument("--labels", default="data/labels/word_to_label.pkl")
    parser.add_argument("--features", default=None, help="Pre-extracted features path")
    parser.add_argument("-r", "--rate", type=int, default=4, help="Frame rate")
    parser.add_argument("--seg", action="store_true", help="Enable segmentation")

    args = parser.parse_args()

    # Paths
    output_dir = os.path.abspath(args.output)
    frames_dir = os.path.join(output_dir, "frames")
    features_file = os.path.join(output_dir, "features.pkl")
    log_dir = os.path.join(output_dir, "logs")

    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    print("Starting sign language detection pipeline...")
    print("Input video:", args.input)
    print("Output directory:", output_dir)

    # ------------------------------
    # Step 1: Video to Frames
    # ------------------------------
    print("\nStep 1: Converting video to frames...")
    run_command(
        [
            sys.executable, "-m", "src.preprocessing.video_to_frames",
            "-i", args.input,
            "-o", frames_dir,
            "-f", str(args.rate)
        ],
        log_file=os.path.join(log_dir, "video_to_frames.log")
    )

    # ------------------------------
    # Step 2: Segmentation (optional)
    # ------------------------------
    if args.seg:
        print("\nStep 2: Applying segmentation...")

        sapiens_dir = r"C:\Users\purvs\Downloads\sapiens\lite\scripts\demo\torchscript"
        seg_script = os.path.join(sapiens_dir, "seg.sh")

        if os.path.exists(seg_script):
            run_command(
                ["bash", seg_script, "-i", frames_dir, "-o", os.path.join(frames_dir, "sap")]
            )

        run_command(
            [
                sys.executable, "-m", "src.segmentation.segment",
                "-i", frames_dir,
                "-l", os.path.join(log_dir, "segmentation.log")
            ],
            log_file=os.path.join(log_dir, "segmentation.log")
        )

        frames_dir = os.path.join(frames_dir, "mask")

    # ------------------------------
    # Step 3: Feature Extraction
    # ------------------------------
    if args.features is None:
        print("\nStep 3: Extracting features...")
        run_command(
            [
                sys.executable, "-m", "src.preprocessing.feature_extraction",
                "-i", frames_dir,
                "-o", features_file,
                "-m", args.model
            ],
            log_file=os.path.join(log_dir, "feature_extraction.log")
        )
    else:
        print("\nStep 3: Using provided features:", args.features)
        features_file = args.features

    # ------------------------------
    # Step 4: Prediction
    # ------------------------------
    print("\nStep 4: Running prediction...")
    run_command(
        [
            sys.executable, "-m", "src.inference.predict",
            "-i", frames_dir
        ]
    )

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
