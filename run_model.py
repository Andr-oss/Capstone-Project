import os
import cv2
import torch
import pandas as pd
import numpy as np
from torch import nn
from torchvision.models import resnet50


class DeepLabCutWrapper(nn.Module):
    def __init__(self, num_keypoints):
        super().__init__()
        # Create backbone with proper layer structure
        self.backbone = resnet50(pretrained=False)
        self.backbone = nn.Sequential(
            self.backbone.conv1,
            self.backbone.bn1,
            self.backbone.relu,
            self.backbone.maxpool,
            self.backbone.layer1,
            self.backbone.layer2,
            self.backbone.layer3,
            self.backbone.layer4
        )

        # Add DLC-style deconvolution heads
        self.heatmap_head = nn.Sequential(
            nn.ConvTranspose2d(2048, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, num_keypoints, kernel_size=4, stride=2, padding=1)
        )

    def forward(self, x):
        features = self.backbone(x)
        heatmaps = self.heatmap_head(features)
        return heatmaps


def process_video_with_model(video_path, model_path, output_dir=None,
                             device='cuda' if torch.cuda.is_available() else 'cpu'):
    # Set up paths
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    os.makedirs(output_dir, exist_ok=True)

    video_name = os.path.basename(video_path).split('.')[0]
    output_csv = os.path.join(output_dir, f"{video_name}_tracking.csv")

    # Initialize model
    num_keypoints = 7
    model = DeepLabCutWrapper(num_keypoints)

    # Load checkpoint with proper key mapping
    checkpoint = torch.load(model_path, map_location=device)

    # Adapt checkpoint keys to match our model structure
    state_dict = {}
    for k, v in checkpoint['model'].items():
        if k.startswith('backbone.model.'):
            new_key = k.replace('backbone.model.', 'backbone.')
            state_dict[new_key] = v
        elif k.startswith('heads.bodypart.heatmap_head.'):
            new_key = k.replace('heads.bodypart.heatmap_head.', 'heatmap_head.')
            state_dict[new_key] = v

    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()

    # Video parameters from config
    crop_params = (0, 240, 0, 240)  # y1, y2, x1, x2
    input_size = 224

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
    # Change these paths as needed
    VIDEO_PATH = r"C:\Users\Bazil\Downloads\f042814_Video.mp4"
    MODEL_PATH = r"snapshot-best-300.pt"

    process_video_with_model(VIDEO_PATH, MODEL_PATH)