from concurrent.futures import process

import deeplabcut
from pathlib import Path
import os

# === CONFIGURATION ===
CONFIG_PATH = r"D:/Capstone-Project/AndrewFirstTraining-Andrew-2025-03-08/config.yaml"
VIDEO_PATH = r"D:/Capstone-Project/AndrewFirstTraining-Andrew-2025-03-08/f042814_Video.avi"
OUTPUT_FOLDER = Path(VIDEO_PATH).parent
SAVE_AS_CSV = True

# === STEP 1: Run inference with DLC ===
print(" Analyzing video with DeepLabCut...")
deeplabcut.analyze_videos(
    config=CONFIG_PATH,
    videos=[VIDEO_PATH],
    save_as_csv=SAVE_AS_CSV,
    destfolder=str(OUTPUT_FOLDER),
    auto_track=False  # Set to True if you have refinements or multiple animals
)

print(" Keypoints extracted.")

# === STEP 2: Find the most recent output CSV ===
import glob
import pandas as pd

print(" Searching for the latest output CSV file...")
csv_files = sorted(glob.glob(str(OUTPUT_FOLDER / "*DLC*.csv")), key=os.path.getmtime, reverse=True)
if not csv_files:
    raise FileNotFoundError(" No DLC output CSV files found!")

latest_csv = csv_files[0]
print(f" Found DLC keypoints file: {latest_csv}")

# === STEP 3: Optionally post-process it (your existing post-process pipeline) ===
from post_process import postprocess_csv  # We can create this wrapper from your logic

output_post_file = str(OUTPUT_FOLDER / "output_postprocessed.csv")
postprocess_csv(latest_csv, output_post_file)

print(f" Done! Final processed file saved at: {output_post_file}")
