import deeplabcut
from pathlib import Path
import os, glob
import pandas as pd
import cv2
from post_process import postprocess_csv


def run_dlc_pipeline(
    video_path: str,
    progress_value,
    final_csv,
    config_path: str = r"D:/Capstone-Project/AndrewFirstTraining-Andrew-2025-03-08/config.yaml"
):

    output_folder = Path(video_path).parent
    save_as_csv = True

    # === STEP 1: Get total frame count first ===
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    # === STEP 2: Run inference frame-by-frame ===
    progress_value.value = 5
    print(f" Starting DLC analysis ({total_frames} frames)...")

    from deeplabcut.utils.auxiliaryfunctions import read_config
    cfg = read_config(config_path)
    video = str(video_path)

    # Monkey-patch tqdm to capture progress
    from tqdm import tqdm
    real_tqdm = tqdm
    real_tqdm = tqdm

    def custom_tqdm(*args, **kwargs):
        bar = real_tqdm(*args, **kwargs)
        original_update = bar.update

        def patched_update(n=1):
            original_update(n)
            percent = int(25 + (bar.n / total_frames) * 50)  # DLC phase = 25–75%
            progress_value.value = min(percent, 75)

        bar.update = patched_update
        return bar

    # Apply the monkey patch
    import deeplabcut.utils.auxiliaryfunctions
    deeplabcut.utils.auxiliaryfunctions.tqdm = custom_tqdm


    deeplabcut.analyze_videos(
        config=config_path,
        videos=[video],
        save_as_csv=save_as_csv,
        destfolder=str(output_folder),
        auto_track=False
    )

    print(" Keypoints extracted.")

    # === STEP 3: Find the latest CSV ===
    progress_value.value = 80
    csv_files = sorted(glob.glob(str(output_folder / "*DLC*.csv")), key=os.path.getmtime, reverse=True)
    if not csv_files:
        progress_value.value = 100
        raise FileNotFoundError("❌ No DLC output CSV files found!")

    latest_csv = csv_files[0]
    print(f" Found CSV: {latest_csv}")

    output_post_file = str(output_folder / "output_postprocessed.csv")
    postprocess_csv(latest_csv, output_post_file)
    print(f" Final processed file saved: {output_post_file}")

    progress_value.value = 100
    final_csv.value = output_post_file
    return output_post_file
