import pandas as pd
import numpy as np

# File path
input_file = r"C:\Users\Bazil\Downloads\CollectedData_Chen (1).csv"
output_file = r"C:\Users\Bazil\Downloads\output.csv"

# Read the CSV file
df = pd.read_csv(input_file, header=[1, 2])  # Read headers

# Flatten headers
df.columns = [f"{col1}_{col2}" if pd.notna(col2) else col1 for col1, col2 in df.columns]

# Rename frame column
df.rename(columns={df.columns[0]: "Frame"}, inplace=True)

print("Columns:", df.columns.tolist())


# Define function to find center of three points
def find_centroid(p1, p2, p3):
    # Check if any value is None or NaN
    if (pd.isna(p1[0]) or pd.isna(p1[1]) or
            pd.isna(p2[0]) or pd.isna(p2[1]) or
            pd.isna(p3[0]) or pd.isna(p3[1])):
        return np.nan, np.nan

    x = round((float(p1[0]) + float(p2[0]) + float(p3[0])) / 3, 2)
    y = round((float(p1[1]) + float(p2[1]) + float(p3[1])) / 3, 2)
    return x, y


def estimate_missing_part(missing_part, current_frame):
    """
    Estimates coordinates for a missing body part based on anatomical relationships.

    Parameters:
    missing_part (str): Name of the missing body part (e.g., 'Right_Ear')
    current_frame (pd.Series): A row from the DataFrame containing body part coordinates

    Returns:
    tuple: (estimated_x, estimated_y)
    """

    # Helper function to get coordinates
    def get_coords(part):
        return current_frame[('Chen', part, 'x')], current_frame[('Chen', part, 'y')]

    match missing_part:
        case 'Right_Ear':
            hx, hy = get_coords('Head')
            lx, ly = get_coords('Left_Ear')
            return 2 * hx - lx, 2 * hy - ly  # Mirror across head
        case 'Left_Ear':
            hx, hy = get_coords('Head')
            rx, ry = get_coords('Right_Ear')
            return 2 * hx - rx, 2 * hy - ry
        case 'Right_Body':
            cx, cy = get_coords('Body_Center')
            lx, ly = get_coords('Left_Body')
            return 2 * cx - lx, 2 * cy - ly
        case 'Left_Body':
            cx, cy = get_coords('Body_Center')
            rx, ry = get_coords('Right_Body')
            return 2 * cx - rx, 2 * cy - ry
        case 'Tail_Base':
            # Estimate based on body center and previous position (simple approach)
            cx, cy = get_coords('Body_Center')
            return cx, cy + 10  # Adjust offset based on your data
        case 'Nose':
            # Average of ears when head is missing
            lx, ly = get_coords('Left_Ear')
            rx, ry = get_coords('Right_Ear')
            return (lx + rx) / 2, (ly + ry) / 2


# Ensure correct column names before applying function
required_columns = ["Head_x", "Head_y", "Left_Ear_x", "Left_Ear_y", "Right_Ear_x", "Right_Ear_y"]
if all(col in df.columns for col in required_columns):

    # Interpolate missing values in the required columns
    for col in required_columns:
        # Convert to numeric, replacing empty strings with NaN
        df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')
        # Use pandas interpolate method to fill single missing values
        df[col] = df[col].interpolate(method='linear')
        df[col] = df[col].round(2)

    # Calculate centroids
    centroids = df.apply(lambda row: find_centroid(
        (row["Head_x"], row["Head_y"]),
        (row["Left_Ear_x"], row["Left_Ear_y"]),
        (row["Right_Ear_x"], row["Right_Ear_y"])
    ), axis=1)

    # Extract x and y coordinates from the series of tuples
    df["Head_Center_x"] = centroids.apply(lambda x: x[0])
    df["Head_Center_y"] = centroids.apply(lambda x: x[1])
else:
    print("Error: Missing required columns", required_columns)

# Round all numerical columns to 2 decimal places
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].round(2)

# Save the processed file
df.to_csv(output_file, index=False)

print(df.head())
print(f"Processed data saved to {output_file}")