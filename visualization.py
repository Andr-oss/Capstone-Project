import pandas as pd
import numpy as np
import cv2


class RodentVisualizerCV:
    def __init__(self, csv_path, video_path=None, width=800, height=600, trail_length=None,
                 show_trails=True, show_connections=True, show_segmentation=True,
                 segmentation_opacity=0.3, show_labels=True, show_legend=True):
        # Load the data
        self.data = pd.read_csv(csv_path)

        # Debug: Print column names to check exact format
        print("CSV columns:", self.data.columns.tolist())

        # Find the frame column (first column)
        self.frame_column = self.data.columns[0]  # Assume first column is frame
        print(f"Using '{self.frame_column}' as frame identifier column")

        # Get unique frame identifiers and count
        self.frame_ids = self.data[self.frame_column].unique()
        self.frames = len(self.frame_ids)
        print(f"Found {self.frames} unique frames")

        # Set up display dimensions
        self.width = width
        self.height = height

        # Visualization toggles
        self.show_trails = show_trails
        self.show_connections = show_connections
        self.show_segmentation = show_segmentation
        self.segmentation_opacity = segmentation_opacity
        self.show_labels = show_labels
        self.show_legend = show_legend
        self.video_path = video_path
        self.has_video = False

        # Detect body parts and track original column names
        self.body_part_columns = {}  # Format: {processed_name: {"_x": "Original_X_col", "_y": "Original_Y_col"}}
        for col in self.data.columns:
            if col.endswith("_x") or col.endswith("_y"):
                # Split into base name and suffix while preserving original case
                base_name, suffix = col.rsplit("_", 1)
                processed_name = base_name.lower().replace(" ", "_")  # Normalize to lowercase snake_case

                # Track original columns
                if processed_name not in self.body_part_columns:
                    self.body_part_columns[processed_name] = {}
                self.body_part_columns[processed_name][f"_{suffix}"] = col  # Store original column name

        # Filter to valid parts with both coordinates
        self.body_parts = [part for part, cols in self.body_part_columns.items() if "_x" in cols and "_y" in cols]
        print(f"Detected body parts: {self.body_parts}")

        # Assign colors to each body part (BGR format for OpenCV)
        self.colors = {}
        color_list = [
            (0, 0, 255),  # Red (Nose)
            (0, 255, 0),  # Green (body_center)
            (255, 0, 0),  # Blue (tail_base)
            (255, 0, 255),  # Magenta (right_ear)
            (0, 255, 255),  # Yellow (left_ear)
            (255, 255, 0),  # Cyan (right_body)
            (128, 0, 255),  # Purple (left_body)
        ]

        for i, part in enumerate(self.body_parts):
            self.colors[part] = color_list[i % len(color_list)]

        # Define segmentation colors
        self.segmentation_colors = {
            'ear': (200, 200, 255),  # Light red (BGR format)
            'body': (200, 255, 255),  # Light yellow
            'tail': (255, 220, 200),  # Light blue
        }

        # Calculate bounds using ACTUAL column names
        x_cols = [cols["_x"] for part, cols in self.body_part_columns.items() if "_x" in cols]
        y_cols = [cols["_y"] for part, cols in self.body_part_columns.items() if "_y" in cols]

        self.min_x = self.data[x_cols].min().min()
        self.max_x = self.data[x_cols].max().max()
        self.min_y = self.data[y_cols].min().min()
        self.max_y = self.data[y_cols].max().max()
        print(f"Data bounds: X={self.min_x} to {self.max_x}, Y={self.min_y} to {self.max_y}")

        #Add some padding to the bounds
        pad_x = (self.max_x - self.min_x) * 0.1
        pad_y = (self.max_y - self.min_y) * 0.1
        self.min_x -= pad_x
        self.max_x += pad_x
        self.min_y -= pad_y
        self.max_y += pad_y

        # Set up the trail data for each part
        self.trails = {part: [] for part in self.body_parts}
        self.trail_length = trail_length  # Number of frames to show in the trail (None = unlimited)

        # Initialize video if path is provided
        if self.video_path:
            self.video_capture = cv2.VideoCapture(self.video_path)
            if self.video_capture.isOpened():
                self.has_video = True
                # Get video properties
                self.video_width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.video_height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
                self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

                print(
                    f"Video loaded: {self.video_width}x{self.video_height}, {self.video_fps} FPS, {self.video_frame_count} frames")

                # Adjust display dimensions to match video if needed
                if self.width == 800 and self.height == 600:  # If default dimensions weren't changed
                    self.width = self.video_width
                    self.height = self.video_height
            else:
                print(f"Warning: Could not open video file {self.video_path}")

    def normalize_coords(self, x, y):
        # Convert data coordinates to pixel coordinates with bounds checking
        norm_x = int((x - self.min_x) / (self.max_x - self.min_x) * (self.width - 40) + 20)
        norm_y = int((y - self.min_y) / (self.max_y - self.min_y) * (self.height - 40) + 20)

        # Ensure within bounds
        norm_x = max(0, min(norm_x, self.width - 1))
        norm_y = max(0, min(norm_y, self.height - 1))

        return norm_x, norm_y


    def display_animation(self, delay=30):

        if self.has_video:
            self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.width = int(self.video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.height = int(self.video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.video_fps = self.video_capture.get(cv2.CAP_PROP_FPS)
            self.video_frame_count = int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

        # Create window
        cv2.namedWindow("Rodent Movement Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rodent Movement Tracking", self.width, self.height)

        processed_frames = 0

        # Process each frame
        for frame_idx, frame_id in enumerate(sorted(self.frame_ids)):
            # Get data for current frame
            frame_data = self.data[self.data[self.frame_column] == frame_id]
            print(f"Processing frame {frame_idx + 1}/{self.frames}: {frame_id}, Data rows: {len(frame_data)}")

            # Create canvas - either from video or blank
            if self.has_video:
                # Try to read corresponding frame from video
                ret, video_frame = self.video_capture.read()
                if not ret:
                    print("Warning: Reached end of video before end of tracking data")
                    # Use last frame or create blank canvas
                    canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255
                else:
                    # Resize video frame if necessary
                    if video_frame.shape[1] != self.width or video_frame.shape[0] != self.height:
                        canvas = cv2.resize(video_frame, (self.width, self.height))
                    else:
                        canvas = video_frame.copy()
            else:
                # Use blank canvas as before
                canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

                # Draw coordinate grid on blank canvas only
                for grid_i in range(0, self.width, 100):
                    cv2.line(canvas, (grid_i, 0), (grid_i, self.height), (240, 240, 240), 1)
                for grid_i in range(0, self.height, 100):
                    cv2.line(canvas, (0, grid_i), (self.width, grid_i), (240, 240, 240), 1)

            # Store positions for connections
            positions = {}

            # Calculate positions for all body parts first
            for part in self.body_parts:
                cols = self.body_part_columns[part]  # Get original column names
                x_col = cols["_x"]
                y_col = cols["_y"]

                x = frame_data[x_col].values
                y = frame_data[y_col].values

                # Check for valid (non-null, non-blank) values
                if len(x) > 0 and len(y) > 0 and not pd.isna(x[0]) and not pd.isna(y[0]):
                    # Normalize coordinates
                    if not self.has_video:
                        px, py = self.normalize_coords(x[0], y[0])
                    else:
                        px, py = int(x[0]), int(y[0])

                    positions[part] = (px, py)
                    # Update trail
                    self.trails[part].append((px, py))
                    # Only limit trail length if trail_length is specified
                    if self.trail_length is not None and len(self.trails[part]) > self.trail_length:
                        self.trails[part].pop(0)

            # Draw segmentation if requested (bottom layer)
            if self.show_segmentation:
                # Create a transparent overlay for the polygon fills
                overlay = canvas.copy()

                # Fill main body area
                if all(part in positions for part in ['left_body', 'body_center', 'right_body', 'tail_base']):
                    if all(part in positions for part in ['left_ear', 'right_ear']):
                        body_polygon = np.array([
                            positions['left_body'],
                            positions['left_ear'],
                            positions['right_ear'],
                            positions['right_body'],
                            positions['tail_base']
                        ], np.int32)
                        cv2.fillPoly(overlay, [body_polygon], self.segmentation_colors['body'])

                # Fill ear-nose triangle
                if all(part in positions for part in ['left_ear', 'nose', 'right_ear']):
                    ear_triangle = np.array([
                        positions['left_ear'],
                        positions['nose'],
                        positions['right_ear']
                    ], np.int32)
                    cv2.fillPoly(overlay, [ear_triangle], self.segmentation_colors['ear'])

                # Blend the overlay with the main canvas
                cv2.addWeighted(overlay, self.segmentation_opacity, canvas, 1 - self.segmentation_opacity, 0, canvas)

            # Draw connections if requested (middle layer)
            if self.show_connections:
                # Connect nose-body_center-tail_base (Make Spine)
                if all(part in positions for part in ['nose', 'body_center', 'tail_base']):
                    cv2.line(canvas, positions['nose'], positions['body_center'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['body_center'], positions['tail_base'], (100, 100, 100), 2)

                # Connect the ears and nose as a triangle
                if all(part in positions for part in ['left_ear', 'nose', 'right_ear']):
                    cv2.line(canvas, positions['left_ear'], positions['nose'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['nose'], positions['right_ear'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['left_ear'], positions['right_ear'], (100, 100, 100), 2)

                # Connect body sides
                if all(part in positions for part in ['left_body', 'body_center', 'right_body']):
                    cv2.line(canvas, positions['left_body'], positions['body_center'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['body_center'], positions['right_body'], (100, 100, 100), 2)

                # Connect body sides to ears
                if all(part in positions for part in ['left_body', 'right_body', 'left_ear', 'right_ear']):
                    cv2.line(canvas, positions['left_body'], positions['left_ear'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['right_body'], positions['right_ear'], (100, 100, 100), 2)

                # Connect body sides to tail base
                if all(part in positions for part in ['left_body', 'right_body', 'tail_base']):
                    cv2.line(canvas, positions['left_body'], positions['tail_base'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['right_body'], positions['tail_base'], (100, 100, 100), 2)

            # Draw trails and points (top layer)
            for part in self.body_parts:
                if part in positions:
                    px, py = positions[part]

                    # Draw trail
                    if self.show_trails:
                        for trail_i in range(1, len(self.trails[part])):
                            if self.trail_length is not None:
                                # Fade based on trail position
                                alpha = 0.3 + 0.7 * trail_i / len(self.trails[part])
                            else:
                                # If unlimited trail, fade based on distance from current point
                                distance_from_current = len(self.trails[part]) - trail_i
                                alpha = max(0.1, 1.0 - (distance_from_current / 50.0))  # Fade out over 50 frames

                            color = self.colors[part]
                            # Scale alpha to color
                            scaled_color = tuple(int(c * alpha) for c in color)
                            cv2.line(canvas, self.trails[part][trail_i - 1], self.trails[part][trail_i], scaled_color,
                                     2)

                    # Draw current position (larger dot)
                    cv2.circle(canvas, (px, py), 3, self.colors[part], -1)

                    # Label the dot
                    if self.show_labels:
                        cv2.putText(canvas, part, (px + 10, py), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[part], 1)

            # Add frame counter
            cv2.putText(canvas, f"Frame: {frame_idx + 1}/{self.frames}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Add legend for body parts
            if self.show_legend:
                legend_y = 60
                for part in self.body_parts:
                    cv2.circle(canvas, (30, legend_y), 6, self.colors[part], -1)
                    cv2.putText(canvas, part, (45, legend_y + 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                    legend_y += 25

            # Display the image
            cv2.imshow("Rodent Movement Tracking", canvas)
            print(f"Displayed frame {frame_idx + 1}")

            processed_frames += 1

            # Wait for delay or key press
            key = cv2.waitKey(delay)
            if key == 27 or key == ord('q'):  # ESC or 'q' to quit
                break
            elif key == 32:  # Space to pause/resume
                cv2.waitKey(0)

        print(f"Total frames processed: {processed_frames}")
        # Wait for a key press before closing
        print("Animation complete, press any key to close")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# Example usage
if __name__ == "__main__":
    # Replace with your actual CSV file path
    input_file = r"C:\Users\mbazi\Downloads\output22.csv"  # Update with your path
    video_file = r"C:\Users\mbazi\Downloads\f042814_Video.mp4"  # Optional video

    # Create visualizer with customizable visualization options
    visualizer = RodentVisualizerCV(
        input_file,
        video_file,
        width=960,
        height=960,
        trail_length=5,  # Show last x frames of trail
        show_trails=False,  # Toggle trails on/off
        show_connections=True,  # Toggle connections on/off
        show_segmentation=True,  # Toggle segmentation fill on/off
        segmentation_opacity=0.5,  # Opacity of segmentation fill (0-1)
        show_labels=False,  # Toggle labels on/off
        show_legend=True  # Toggle legend on/off
    )
    visualizer.display_animation(delay=60)  # delay for debugging



