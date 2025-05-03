import sys
import os
import pathlib
import signal
import json
import atexit
import cv2
import time
import numpy as np

# Add the parent directory to sys.path
BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.append(str(BASE_DIR))

from tracking import visualization

def write_status(status):
    """Write status to a temporary file that Django can read"""
    status_file = os.path.join(BASE_DIR, 'temp_visualizations', 'vis_status.json')
    try:
        with open(status_file, 'w') as f:
            json.dump({'status': status}, f)
    except:
        pass

def run_visualization_to_client(csv_path, video_path=None,
                               trail_length=10, show_trails=False, show_trajectory=True,
                               trajectory_opacity=0.3, show_connections=True, show_segmentation=True,
                               segmentation_opacity=0.5, show_labels=True, show_legend=True,
                               output_file=None, fps=30):
    """
    Run the visualization logic and write the frames to an MP4 video.
    Returns the path to the generated video file.
    """
    # Create the visualizer instance using your existing class.
    visualizer = visualization.RodentVisualizerCV(
        csv_path=csv_path,
        video_path=video_path,
        trail_length=trail_length,
        show_trails=show_trails,
        show_trajectory=show_trajectory,
        trajectory_opacity=trajectory_opacity,
        show_connections=show_connections,
        show_segmentation=show_segmentation,
        segmentation_opacity=segmentation_opacity,
        show_labels=show_labels,
        show_legend=show_legend
    )

    # Decide the output file location if not provided.
    if output_file is None:
        temp_dir = os.path.join(BASE_DIR, 'temp_visualizations')
        os.makedirs(temp_dir, exist_ok=True)
        output_file = os.path.join(temp_dir, f'visualization_{int(time.time())}.mp4')

    # Set up VideoWriter. Choose codec 'mp4v' for MP4.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (visualizer.width, visualizer.height))

    # If a video file was provided, reset to beginning.
    if visualizer.video_path and visualizer.video_capture.isOpened():
        visualizer.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Loop through each frame as in your display_animation logic.
    for frame_idx, frame_id in enumerate(sorted(visualizer.frame_ids)):
        # Get data for the current frame
        frame_data = visualizer.data[visualizer.data[visualizer.frame_column] == frame_id]

        # Create canvas:
        if visualizer.video_path and visualizer.video_capture.isOpened():
            ret, video_frame = visualizer.video_capture.read()
            if not ret:
                break  # End if video runs out
            canvas = video_frame.copy()
        else:
            canvas = 255 * np.ones((visualizer.height, visualizer.width, 3), dtype=np.uint8)
            # Optionally draw gridlines or background elements here

        # (Optional) Draw overlays such as frame count:
        cv2.putText(canvas, f"Frame: {frame_idx+1}/{visualizer.frames}", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

        # [Place here additional drawing routines copied from your display_animation() code if desired.]

        # Write the frame to the video.
        out.write(canvas)

        # Optionally, update a status file (if desired) and add a small delay to mimic processing time.
        # time.sleep(1/fps)

    # Release the writer when finished.
    out.release()
    write_status('closed')
    return output_file

