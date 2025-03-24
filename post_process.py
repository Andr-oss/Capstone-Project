import pandas as pd
import numpy as np

# File paths
input_file = r"Macintosh HD\Users\lionkk\Downloads\CollectedData_Chen.csv"
output_file = r"Macintosh HD\Users\lionkk\Downloads\ProcessedData_Chen.csv"

# Read the CSV file with multi-level headers
df = pd.read_csv(input_file, header=[1, 2])

# Make a copy of the original multi-index columns for reference
original_columns = df.columns.tolist()

# Flatten the multi-level column headers
df.columns = [f"{col1}_{col2}" if pd.notna(col2) else col1 for col1, col2 in df.columns]

# Rename first column to "Frame"
df.rename(columns={df.columns[0]: "Frame"}, inplace=True)

print("Columns:", df.columns.tolist())


# Function to calculate centroid of three points
def find_centroid(p1, p2, p3):
    # Check if any coordinates are missing
    if (pd.isna(p1[0]) or pd.isna(p1[1]) or
            pd.isna(p2[0]) or pd.isna(p2[1]) or
            pd.isna(p3[0]) or pd.isna(p3[1])):
        return np.nan, np.nan

    # Calculate centroid coordinates
    x = round((float(p1[0]) + float(p2[0]) + float(p3[0])) / 3, 2)
    y = round((float(p1[1]) + float(p2[1]) + float(p3[1])) / 3, 2)
    return x, y


# Function to estimate missing body parts based on anatomical relationships
def estimate_missing_part(part, row):
    # Check which part we're estimating
    if part == 'Right_Ear':
        if not (pd.isna(row["Nose_x"]) or pd.isna(row["Left_Ear_x"])):
            # Mirror Left Ear across Nose
            return (2 * row["Nose_x"] - row["Left_Ear_x"],
                    2 * row["Nose_y"] - row["Left_Ear_y"])

    elif part == 'Left_Ear':
        if not (pd.isna(row["Nose_x"]) or pd.isna(row["Right_Ear_x"])):
            # Mirror Right Ear across Nose
            return (2 * row["Nose_x"] - row["Right_Ear_x"],
                    2 * row["Nose_y"] - row["Right_Ear_y"])

    elif part == 'Nose':
        if not (pd.isna(row["Left_Ear_x"]) or pd.isna(row["Right_Ear_x"])):
            # Average of ears
            return ((row["Left_Ear_x"] + row["Right_Ear_x"]) / 2,
                    (row["Left_Ear_y"] + row["Right_Ear_y"]) / 2)

    elif part == 'Right_Body':
        if not (pd.isna(row["Body_Center_x"]) or pd.isna(row["Left_Body_x"])):
            # Mirror Left Body across Body Center
            return (2 * row["Body_Center_x"] - row["Left_Body_x"],
                    2 * row["Body_Center_y"] - row["Left_Body_y"])

    elif part == 'Left_Body':
        if not (pd.isna(row["Body_Center_x"]) or pd.isna(row["Right_Body_x"])):
            # Mirror Right Body across Body Center
            return (2 * row["Body_Center_x"] - row["Right_Body_x"],
                    2 * row["Body_Center_y"] - row["Right_Body_y"])

    elif part == 'Tail_Base':
        if not pd.isna(row["Body_Center_x"]):
            # Estimate based on body center
            return (row["Body_Center_x"], row["Body_Center_y"] + 10)

    # Return NaN if estimation is not possible
    return np.nan, np.nan


# Process all body part columns
body_parts = ["Nose", "Left_Ear", "Right_Ear", "Body_Center", "Left_Body", "Right_Body", "Tail_Base"]

# First, convert all coordinate columns to numeric
for part in body_parts:
    for suffix in ['_x', '_y']:
        col = f"{part}{suffix}"
        if col in df.columns:
            df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')

# Apply basic interpolation first
for part in body_parts:
    for suffix in ['_x', '_y']:
        col = f"{part}{suffix}"
        if col in df.columns:
            # Interpolate linearly where possible
            df[col] = df[col].interpolate(method='linear', limit=3)  # Limit consecutive gaps

# Then estimate missing values that couldn't be interpolated
for idx, row in df.iterrows():
    for part in body_parts:
        x_col = f"{part}_x"
        y_col = f"{part}_y"

        if x_col in df.columns and y_col in df.columns:
            if pd.isna(row[x_col]) or pd.isna(row[y_col]):
                # Try to estimate the missing coordinates
                est_x, est_y = estimate_missing_part(part, row)

                if not (pd.isna(est_x) or pd.isna(est_y)):
                    df.at[idx, x_col] = est_x
                    df.at[idx, y_col] = est_y

# Calculate head centroids for each row using Nose and Ears
df["Head_Center_x"] = np.nan
df["Head_Center_y"] = np.nan

for idx, row in df.iterrows():
    if not (pd.isna(row["Nose_x"]) or pd.isna(row["Left_Ear_x"]) or pd.isna(row["Right_Ear_x"])):
        x, y = find_centroid(
            (row["Nose_x"], row["Nose_y"]),
            (row["Left_Ear_x"], row["Left_Ear_y"]),
            (row["Right_Ear_x"], row["Right_Ear_y"])
        )
        df.at[idx, "Head_Center_x"] = x
        df.at[idx, "Head_Center_y"] = y

# Round all numerical columns to 2 decimal places
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].round(2)

# Save processed data
df.to_csv(output_file, index=False)

print(df.head())
print(f"Processed data saved to {output_file}")