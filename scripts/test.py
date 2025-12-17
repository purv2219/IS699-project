import os
import sys
import argparse
import subprocess

# ------------------------------
# Sign Language Inference Script (Fixed Paths)
# ------------------------------

def run_cmd(cmd, cwd=None):
    """Run a command and stop if it fails."""
    print(f"[{' '.join(cmd)}] Running...")
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing command: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="Sign Language Pipeline")

    parser.add_argument("-i", "--input_dir", 
                        default=r"C:\Users\purvs\Downloads\png-segments",
                        help="Directory containing frame folders")
    
    parser.add_argument("-o", "--output_dir", default="output_inference", help="Directory to save logs/predictions")
    parser.add_argument("--use_segmentation", action="store_true", help="Enable Sapiens segmentation")
    parser.add_argument("--sapiens_path", 
                        default=r"C:\Users\purvs\Downloads\sapiens\lite\scripts\demo\torchscript",
                        help="Path to Sapiens torchscript folder")

    args = parser.parse_args()

    # Define Paths
    base_dir = os.path.abspath(args.input_dir)
    output_dir = os.path.abspath(args.output_dir)
    log_dir = os.path.join(output_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    # Project Root (Assumed to be current working directory)
    project_root = os.getcwd()

    print("=== Starting Pipeline ===")
    print(f"Data Source: {base_dir}")

    # -------------------------------------------------
    # Step 1: Segmentation (Optional)
    # -------------------------------------------------
    if args.use_segmentation:
        print("\n=== Step 1: Running Segmentation ===")
        subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
        if not subfolders: subfolders = [base_dir]

        seg_script = "seg.sh" 

        for folder in subfolders:
            folder_name = os.path.basename(folder)
            try:
                subprocess.run([
                    "bash", seg_script,
                    "-i", folder,
                    "-o", os.path.join(folder, "sap")
                ], cwd=args.sapiens_path, check=True)
            except Exception as e:
                print(f"Warning: Sapiens failed for {folder_name}: {e}")
                continue

            run_cmd([
                sys.executable, "-m", "src.segmentation.segment",
                "-i", folder,
                "-l", os.path.join(log_dir, "segmentation.log")
            ])
    else:
        print("\nSkipping Segmentation")

    # -------------------------------------------------
    # Step 2: Feature Extraction
    # -------------------------------------------------
    print("\n=== Step 2: Extracting Features ===")
    
    run_cmd([
        sys.executable, "-m", "src.preprocessing.feature_extraction",
        "-i", base_dir,
        "-t", "test"
    ])
    
    # FIX: Point to the actual location where the script saves features
    # Based on your log: "features/resnet_features_test.pkl"
    features_file = os.path.join(project_root, "features", "resnet_features_test.pkl")
    
    if not os.path.exists(features_file):
        # Fallback check: sometimes it might save to "data/features"
        alt_path = os.path.join(project_root, "data", "features", "resnet_features_test.pkl")
        if os.path.exists(alt_path):
            features_file = alt_path
        else:
            print(f"Error: Features file not found at {features_file}")
            print("Please check where 'src.preprocessing.feature_extraction' is saving the .pkl file.")
            sys.exit(1)

    print(f"Found features at: {features_file}")

    # -------------------------------------------------
    # Step 3: Prediction
    # -------------------------------------------------
    print("\n=== Step 3: Running Prediction ===")
    
    prediction_output = os.path.join(output_dir, "prediction.txt")
    
    # Command uses -i for the input features file
    cmd = [
        sys.executable, "-m", "src.inference.predict",
        "-i", features_file
    ]

    print(f"[{' '.join(cmd)}] Running...")
    
    # result = subprocess.run(cmd, capture_output=True, text=True)

    # if result.returncode != 0:
    #     print("Error in prediction script:")
    #     print(result.stderr)
    #     sys.exit(result.returncode)
    # else:
    #     print("Prediction successful.")
    #     # Save output to file
    #     with open(prediction_output, "w") as f:
    #         f.write(result.stdout)
        
    #     print(f"\nPipeline completed. Results saved to: {prediction_output}")
    #     print("Prediction Result:\n", result.stdout.strip())

if __name__ == "__main__":
    main()