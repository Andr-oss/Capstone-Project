import pandas as pd
import numpy as np

# File path
input_file = r"C:\Users\mbazi\Downloads\CollectedData_Chen.csv"
output_file = r"C:\Users\mbazi\Downloads\output(in).csv"

# Read the CSV file
df = pd.read_csv(input_file, header=[1, 2])  # Read headers

# Flatten headers
df.columns = [f"{col1}_{col2}" if pd.notna(col2) else col1 for col1, col2 in df.columns]

# Drop the first two original header rows and reset index
df = df.iloc[2:].reset_index(drop=True)

# Drop first two columns
df = df.iloc[:, 2:]

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
    df["Processed_Head_Center_x"] = centroids.apply(lambda x: x[0])
    df["Processed_Head_Center_y"] = centroids.apply(lambda x: x[1])
else:
    print("Error: Missing required columns", required_columns)

# Round all numerical columns to 2 decimal places
numeric_columns = df.select_dtypes(include=[np.number]).columns
df[numeric_columns] = df[numeric_columns].round(2)

# Save the processed file
df.to_csv(output_file, index=False)

print(df.head())
print(f"Processed data saved to {output_file}")