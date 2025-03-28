import os
import cv2
import torch
import pandas as pd
import numpy as np
from torchvision import transforms


def process_video_with_model(video_path, model_path, output_dir=None,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Process a video using a pre-trained PyTorch model and save body part coordinates to CSV.

    Args:
        video_path (str): Path to the input video
        model_path (str): Path to the snapshot.pt file
        output_dir (str, optional): Directory to save results. Defaults to video directory.
        device (str): Device to run model on ('cuda' or 'cpu')

    Returns:
        str: Path to the output CSV file
    """
    # Set up output path
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).split('.')[0]
    output_csv = os.path.join(output_dir, f"{video_name}_tracking.csv")

    # Load model - handle dictionary format
    checkpoint = torch.load(model_path, map_location=device)

    # Print keys to understand the structure (remove this in production)
    print("Checkpoint keys:", checkpoint.keys())

    # Try different approaches to get the model based on common formats
    if isinstance(checkpoint, dict):
        if 'model' in checkpoint:
            model = checkpoint['model']
        elif 'state_dict' in checkpoint:
            # This requires knowing your model architecture
            # For demonstration, we'll assume a common DLC architecture
            from torchvision.models import resnet50
            model = resnet50()
            model.load_state_dict(checkpoint['state_dict'])
        elif 'network' in checkpoint:
            model = checkpoint['network']
        else:
            # Try to find the first item that could be a model
            for key, value in checkpoint.items():
                if hasattr(value, 'eval'):
                    model = value
                    break
            else:
                raise ValueError(f"Could not find model in checkpoint. Keys: {list(checkpoint.keys())}")
    else:
        model = checkpoint

    # Set model to evaluation mode
    model.eval()

    # Define the body parts being tracked
    body_parts = ['nose', 'left_ear', 'right_ear']

    # Define image transformations (adjust as needed for your model)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),  # Adjust size to match your model's expected input
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Open video
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Create DataFrame for results
    columns = ['frame']
    for bp in body_parts:
        columns.extend([f"{bp}_x", f"{bp}_y", f"{bp}_confidence"])
    results = pd.DataFrame(columns=columns)

    # Process each frame
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess the frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(frame_rgb).unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            predictions = model(input_tensor)

        # Debug: print the output format to understand structure
        if frame_idx == 0:
            print("Model output type:", type(predictions))
            if isinstance(predictions, tuple):
                print("Output is a tuple with lengths:",
                      [len(p) if hasattr(p, '__len__') else 'scalar' for p in predictions])
            elif isinstance(predictions, dict):
                print("Output is a dict with keys:", predictions.keys())
            elif hasattr(predictions, 'shape'):
                print("Output tensor shape:", predictions.shape)

        # Process the predictions based on model output format
        # You'll need to adapt this part based on the actual output format
        keypoints = []

        # Example conversion - adjust based on your model's actual output
        if isinstance(predictions, torch.Tensor):
            # Assuming output is [batch, num_keypoints, 3] where 3 is [x, y, confidence]
            if len(predictions.shape) == 3 and predictions.shape[2] == 3:
                keypoints = predictions[0].cpu().numpy()
            # Or if it's a heatmap [batch, num_keypoints, height, width]
            elif len(predictions.shape) == 4:
                for i in range(predictions.shape[1]):
                    heatmap = predictions[0, i].cpu().numpy()
                    # Get the coordinates of the maximum activation
                    flat_idx = np.argmax(heatmap)
                    y, x = np.unravel_index(flat_idx, heatmap.shape)
                    conf = heatmap[y, x]
                    # Scale to original image coordinates
                    x = x * width / heatmap.shape[1]
                    y = y * height / heatmap.shape[0]
                    keypoints.append([x, y, conf])
                keypoints = np.array(keypoints)

        # Create a row for this frame
        row_data = [frame_idx]
        for i, bp in enumerate(body_parts):
            if i < len(keypoints):
                x, y, conf = keypoints[i]
                row_data.extend([x, y, conf])
            else:
                row_data.extend([0, 0, 0])  # Default values if keypoint not found

        # Add to results
        results.loc[len(results)] = row_data

        # Print progress
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx / frame_count * 100:.1f}%)")

        frame_idx += 1

    cap.release()

    # Save results
    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")

    return output_csv


if __name__ == "__main__":
    # Example usage
    VIDEO_PATH = r"C:\Users\mbazi\Downloads\P20221101_Video.mp4"
    MODEL_PATH = r"snapshot-best-300.pt"

    process_video_with_model(VIDEO_PATH, MODEL_PATH)