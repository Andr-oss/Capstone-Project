import pandas as pd
import numpy as np
import cv2


class RodentVisualizerCV:
    def __init__(self, csv_path, width=800, height=600, trail_length=None):

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

        # Detect body parts from CSV column names
        self.body_parts = set()
        for col in self.data.columns:
            if col.endswith('_x') or col.endswith('_y'):
                self.body_parts.add(col.rsplit('_', 1)[0])
        self.body_parts = list(self.body_parts)
        print(f"Detected body parts: {self.body_parts}")

        # Assign colors to each body part (BGR format for OpenCV)
        self.colors = {}
        color_list = [
            (0, 0, 255),  # Red (head)
            (0, 255, 0),  # Green (body_center)
            (255, 0, 0),  # Blue (tail_base)
            (255, 0, 255),  # Magenta (right_ear)
            (0, 255, 255),  # Yellow (left_ear)
            (255, 255, 0),  # Cyan (right_body)
            (128, 0, 255),  # Purple (left_body)
        ]

        for i, part in enumerate(self.body_parts):
            self.colors[part] = color_list[i % len(color_list)]

        # Calculate the data bounds to normalize coordinates
        x_cols = [f"{part}_x" for part in self.body_parts]
        y_cols = [f"{part}_y" for part in self.body_parts]

        self.min_x = self.data[x_cols].min().min()
        self.max_x = self.data[x_cols].max().max()
        self.min_y = self.data[y_cols].min().min()
        self.max_y = self.data[y_cols].max().max()
        print(f"Data bounds: X={self.min_x} to {self.max_x}, Y={self.min_y} to {self.max_y}")

        # Add some padding to the bounds
        pad_x = (self.max_x - self.min_x) * 0.1
        pad_y = (self.max_y - self.min_y) * 0.1
        self.min_x -= pad_x
        self.max_x += pad_x
        self.min_y -= pad_y
        self.max_y += pad_y

        # Set up the trail data for each part
        self.trails = {part: [] for part in self.body_parts}
        self.trail_length = trail_length  # Number of frames to show in the trail (None = unlimited)

    def normalize_coords(self, x, y):
        """Convert data coordinates to pixel coordinates with bounds checking"""
        norm_x = int((x - self.min_x) / (self.max_x - self.min_x) * (self.width - 40) + 20)
        norm_y = int((y - self.min_y) / (self.max_y - self.min_y) * (self.height - 40) + 20)

        # Ensure within bounds
        norm_x = max(0, min(norm_x, self.width - 1))
        norm_y = max(0, min(norm_y, self.height - 1))

        return norm_x, norm_y

    def display_animation(self, delay=33, connections=True):

        # Create window
        cv2.namedWindow("Rodent Movement Tracking", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Rodent Movement Tracking", self.width, self.height)

        processed_frames = 0

        # Process each frame
        for frame_idx, frame_id in enumerate(sorted(self.frame_ids)):
            # Get data for current frame
            frame_data = self.data[self.data[self.frame_column] == frame_id]
            print(f"Processing frame {frame_idx + 1}/{self.frames}: {frame_id}, Data rows: {len(frame_data)}")

            # Create a blank canvas
            canvas = np.ones((self.height, self.width, 3), dtype=np.uint8) * 255

            # Draw coordinate grid
            for grid_i in range(0, self.width, 100):
                cv2.line(canvas, (grid_i, 0), (grid_i, self.height), (240, 240, 240), 1)
            for grid_i in range(0, self.height, 100):
                cv2.line(canvas, (0, grid_i), (self.width, grid_i), (240, 240, 240), 1)

            # Store positions for connections
            positions = {}

            # Draw each body part
            for part in self.body_parts:
                x_col = f"{part}_x"
                y_col = f"{part}_y"

                if x_col in frame_data.columns and y_col in frame_data.columns:
                    x = frame_data[x_col].values
                    y = frame_data[y_col].values

                    if len(x) > 0 and len(y) > 0:
                        # Debug: Print coordinate transformation
                        print(f"  Part {part}: Original ({x[0]}, {y[0]})")

                        # Store position for later use in connections
                        px, py = self.normalize_coords(x[0], y[0])
                        print(f"  Part {part}: Normalized ({px}, {py})")

                        positions[part] = (px, py)

                        # Update trail
                        self.trails[part].append((px, py))
                        # Only limit trail length if trail_length is specified
                        if self.trail_length is not None and len(self.trails[part]) > self.trail_length:
                            self.trails[part].pop(0)

                        # Draw trail
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
                        cv2.circle(canvas, (px, py), 6, self.colors[part], -1)

                        # Label the dot
                        cv2.putText(canvas, part, (px + 10, py),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors[part], 1)

            # Draw connections between parts if requested
            if connections:
                # Connect head-body-tail
                if all(part in positions for part in ['head', 'body_center', 'tail_base']):
                    cv2.line(canvas, positions['head'], positions['body_center'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['body_center'], positions['tail_base'], (100, 100, 100), 2)

                # Connect ears to head
                if all(part in positions for part in ['left_ear', 'head', 'right_ear']):
                    cv2.line(canvas, positions['left_ear'], positions['head'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['head'], positions['right_ear'], (100, 100, 100), 2)

                # Connect body sides
                if all(part in positions for part in ['left_body', 'body_center', 'right_body']):
                    cv2.line(canvas, positions['left_body'], positions['body_center'], (100, 100, 100), 2)
                    cv2.line(canvas, positions['body_center'], positions['right_body'], (100, 100, 100), 2)

            # Add frame counter
            cv2.putText(canvas, f"Frame: {frame_idx + 1}/{self.frames}", (20, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

            # Add legend for body parts
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
    csv_path = r"C:\Users\Bazil\Downloads\output.csv"  # Update with your path

    # Create visualizer and display animation
    visualizer = RodentVisualizerCV(csv_path, width=1000, height=800)
    visualizer.display_animation(delay=100)  # Increased delay for debugging