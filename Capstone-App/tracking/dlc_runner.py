import deeplabcut
from pathlib import Path
import os
import glob
import pandas as pd
from .post_process import postprocess_csv


def run_dlc_pipeline(video_path: str,
                     config_path: str = r"C:\Users\Bazil\Downloads\AndrewFirstTraining-Andrew-2025-03-08\config.yaml",
                     output_directory: str = None):
    """
    Runs DeepLabCut analysis and post-processing for a given video.
    Parameters:
        video_path (str):  path to the video file.
        config_path (str): Path to the DeepLabCut config.yaml file.
        output_directory (str): Optional custom output directory for saving results.
    """
    print("🧠 Starting DLC inference...")

    # Use custom output directory if provided, otherwise use video directory
    if output_directory:
        output_folder = Path(output_directory)
        os.makedirs(output_folder, exist_ok=True)
    else:
        output_folder = Path(video_path).parent

    save_as_csv = True

    video_basename = Path(video_path).stem

    # STEP 1: Analyze the video
    print("🔍 Analyzing video with DeepLabCut...")
    deeplabcut.analyze_videos(
        config=config_path,
        videos=[video_path],
        save_as_csv=save_as_csv,
        destfolder=str(output_folder),
        auto_track=False
    )
    print("✅ Keypoints extracted.")

    # STEP 2: Find the output CSV
    print("📂 Searching for the latest output CSV file...")
    csv_files = sorted(glob.glob(str(output_folder / f"*{video_basename}*DLC*.csv")), key=os.path.getmtime,
                       reverse=True)
    if not csv_files:
        # Try looking for any DLC csv if specific video name doesn't match
        csv_files = sorted(glob.glob(str(output_folder / "*DLC*.csv")), key=os.path.getmtime, reverse=True)

    if not csv_files:
        raise FileNotFoundError("❌ No DLC output CSV files found!")

    latest_csv = csv_files[0]
    print(f"✅ Found DLC keypoints file: {latest_csv}")

    # STEP 3: Post-process the CSV
    output_post_file = str(output_folder / f"{video_basename}_processed.csv")
    postprocess_csv(latest_csv, output_post_file)
    print(f"✅ Final processed file saved at: {output_post_file}")

    return output_post_file