import torch
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import csv

# Define the keypoints (order must match your model output)
keypoints = ['nose', 'left_ear', 'right_ear', 'tail_end']

# Load your model from the .pt file.
model_path = r'C:\Users\Bazil\Downloads\AndrewFirstTraining-Andrew-2025-03-08 (1)\dlc-models-pytorch\iteration-0\Capstone-project-DLCMar8-trainset95shuffle1\train\snapshot-best-300.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.load(model_path, map_location=device)
model.eval()

# Define the image transform.
# Make sure these transforms match the ones used during training.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Load and preprocess the image
image_path = 'path/to/your/image.jpg'
img = Image.open(image_path).convert('RGB')
input_tensor = transform(img).unsqueeze(0)  # add batch dimension

# Run inference
with torch.no_grad():
    output = model(input_tensor)

# Assume the model outputs a flat tensor of shape (1, 8):
# [x_nose, y_nose, x_left_ear, y_left_ear, x_right_ear, y_right_ear, x_tail_end, y_tail_end]
preds = output.squeeze(0).cpu().numpy()

# Organize the predictions into a dictionary mapping keypoint names to (x, y) coordinates.
predicted_keypoints = {}
for i, key in enumerate(keypoints):
    x = preds[2 * i]
    y = preds[2 * i + 1]
    predicted_keypoints[key] = (x, y)

print("Predicted Keypoints:")
for k, v in predicted_keypoints.items():
    print(f"{k}: {v}")

# Write the results to a CSV file
csv_filename = "predictions.csv"
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["keypoint", "x", "y"])
    for key, (x, y) in predicted_keypoints.items():
        writer.writerow([key, x, y])

print(f"Results written to {csv_filename}")
