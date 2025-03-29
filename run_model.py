import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch import nn
from torchvision.models import resnet50


class DeepLabCutResNet(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        self.base = nn.Sequential(*list(resnet50(pretrained=False).children())[:-2])
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, num_keypoints, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        x = self.base(x)
        x = self.deconv_layers(x)
        return x


def process_video_with_model(video_path, model_path, output_dir=None,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Set up paths
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).split('.')[0]
    output_csv = os.path.join(output_dir, f"{video_name}_tracking.csv")

    # Initialize model
    num_keypoints = 7  # Based on your body parts
    model = DeepLabCutResNet(num_keypoints)

    # Load trained weights
    checkpoint = torch.load(model_path, map_location=device)
    if 'model' in checkpoint:  # Handle DLC checkpoint format
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    model = model.to(device)
    model.eval()

    # Video parameters (adjust crop as needed)
    crop_params = (0, 240, 0, 240)  # y1, y2, x1, x2 from your config
    input_size = 224  # Standard for ResNet

    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Failed to open video at {video_path}")

    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create results dataframe
    body_parts = ["Nose", "Left_Ear", "Right_Ear", "Body_Center",
                  "Left_Body", "Right_Body", "Tail_Base"]
    columns = ['frame'] + [f"{bp}_{m}" for bp in body_parts for m in ['x', 'y', 'confidence']]
    results = pd.DataFrame(columns=columns)

    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply cropping and preprocessing
        y1, y2, x1, x2 = crop_params
        cropped = frame[y1:y2, x1:x2]
        resized = cv2.resize(cropped, (input_size, input_size))

        # Convert to tensor
        frame_tensor = torch.from_numpy(resized).float().permute(2, 0, 1)
        frame_tensor = frame_tensor / 255.0
        frame_tensor = frame_tensor.unsqueeze(0).to(device)

        # Run inference
        with torch.no_grad():
            heatmaps = model(frame_tensor).squeeze().cpu().numpy()

        # Process heatmaps
        row_data = [frame_idx]
        for i in range(num_keypoints):
            hm = heatmaps[i]
            y_hm, x_hm = np.unravel_index(np.argmax(hm), hm.shape)
            conf = hm[y_hm, x_hm]

            # Convert to original coordinates
            x = (x_hm / hm.shape[1]) * (x2 - x1) + x1
            y = (y_hm / hm.shape[0]) * (y2 - y1) + y1

            row_data.extend([x, y, conf])

        results.loc[len(results)] = row_data

        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx}/{frame_count} frames ({frame_idx / frame_count * 100:.1f}%)")

        frame_idx += 1

    cap.release()
    results.to_csv(output_csv, index=False)
    print(f"Results saved to {output_csv}")
    return output_csv


if __name__ == "__main__":
    # Simply change these paths to match your files
    VIDEO_PATH = r"C:\Users\mbazi\Downloads\P20221101_Video.mp4"
    MODEL_PATH = r"snapshot-best-300.pt"

    process_video_with_model(VIDEO_PATH, MODEL_PATH)