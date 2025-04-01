# tracking/dlc_runner.py
import deeplabcut
from pathlib import Path
import os
import glob
import pandas as pd
from post_process import postprocess_csv

# Import the shared progress variables from progress_manager
from tracking.progress_manager import progress_value, final_csv

def run_dlc_pipeline(
    video_path: str,
    config_path: str = r"D:/Capstone-Project/AndrewFirstTraining-Andrew-2025-03-08/config.yaml"
):
    global progress_value, final_csv

    # 1. Start
    progress_value = 5
    print("🧠 Starting DLC inference...")

    output_folder = Path(video_path).parent
    save_as_csv = True

    # 2. Analyze the video (DLC)
    progress_value = 25
    print("🔍 Analyzing video with DeepLabCut...")
    deeplabcut.analyze_videos(
        config=config_path,
        videos=[video_path],
        save_as_csv=save_as_csv,
        destfolder=str(output_folder),
        auto_track=False
    )
    print("✅ Keypoints extracted.")

    # 3. Search for DLC output
    progress_value = 50
    print("📂 Searching for the latest output CSV file...")
    csv_files = sorted(glob.glob(str(output_folder / "*DLC*.csv")), key=os.path.getmtime, reverse=True)
    if not csv_files:
        progress_value = 100
        raise FileNotFoundError("❌ No DLC output CSV files found!")

    latest_csv = csv_files[0]
    print(f"✅ Found DLC keypoints file: {latest_csv}")

    # 4. Post-process
    progress_value = 75
    output_post_file = str(output_folder / "output_postprocessed.csv")
    postprocess_csv(latest_csv, output_post_file)
    print(f"✅ Final processed file saved at: {output_post_file}")

    # 5. Done
    progress_value = 100
    final_csv = output_post_file  # Provide final CSV to get_progress
    return output_post_file
