import sys
import os
import pathlib
import signal
import json
import atexit
import cv2

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


def cleanup_files():
    """Clean up temporary files"""
    try:
        temp_dir = os.path.join(BASE_DIR, 'temp_visualizations')
        for old_file in os.listdir(temp_dir):
            try:
                file_path = os.path.join(temp_dir, old_file)
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except:
                pass
    except:
        pass


def cleanup(status='closed'):
    """Cleanup without exit"""
    cv2.destroyAllWindows()
    write_status(status)
    cleanup_files()


def cleanup_and_exit(status='closed'):
    """Cleanup and exit"""
    cleanup(status)
    sys.exit(0)


def run_visualization(csv_path, video_path=None, trail_length=None,
                 show_trails=False, show_trajectory=True, trajectory_opacity=0.3, show_connections=True, show_segmentation=True,
                 segmentation_opacity=0.5, show_labels=True, show_legend=True):
    """Helper function to run visualization in a separate process"""
    # Register cleanup function to run on normal exit
    atexit.register(cleanup)

    try:
        visualizer = visualization.RodentVisualizerCV(
            csv_path=csv_path,
            video_path=video_path,
            show_trails=show_trails,
            show_trajectory=show_trajectory,
            trail_length=trail_length,
            show_labels=show_labels,
            show_legend=show_legend,
            trajectory_opacity=trajectory_opacity,
            show_connections=show_connections,
            show_segmentation=show_segmentation,
            segmentation_opacity=segmentation_opacity,
        )

        # Create window with specific properties
        cv2.namedWindow("Rodent Movement Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rodent Movement Tracking", visualizer.width, visualizer.height)

        write_status('running')

        while True:
            frame = visualizer.display_animation(delay=30)
            if frame is None:  # End of visualization
                cleanup('closed')
                break

            # Check for ESC key (27)
            key = cv2.waitKey(30) & 0xFF  # Use & 0xFF for compatibility
            if key == 27:  # ESC pressed
                cleanup('esc_pressed')
                break

    except Exception as e:
        print(f"Visualization error: {str(e)}")
        cleanup('error')
    finally:
        # Only exit if we're the main process
        if __name__ == '__main__':
            sys.exit(0)


if __name__ == '__main__':
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda x, y: cleanup_and_exit('interrupted'))

    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
        run_visualization(csv_path)