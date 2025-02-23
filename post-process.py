import pandas as pd

# File path
input_file = r"C:\Users\Bazil\Downloads\CollectedData_Andrew.csv"
output_file = r"C:\Users\Bazil\Downloads\output.csv"

# Read the CSV file
df = pd.read_csv(input_file, header=[1, 2]) # Read headers

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
    return (p1[0] + p2[0] + p3[0]) / 3, (p1[1] + p2[1] + p3[1]) / 3

# Ensure correct column names before applying function
required_columns = ["Head_x", "Head_y", "Left_Ear_x", "Left_Ear_y", "Right_Ear_x", "Right_Ear_y"]
if all(col in df.columns for col in required_columns):
    df["Processed_Head_Center"] = df.apply(lambda row: find_centroid(
        (float(row["Head_x"]), float(row["Head_y"])),
        (float(row["Left_Ear_x"]), float(row["Left_Ear_y"])),
        (float(row["Right_Ear_x"]), float(row["Right_Ear_y"]))
    ), axis=1)
else:
    print("Error: Missing required columns", required_columns)

# Save the processed file
df.to_csv(output_file, index=False)

print(f"Processed data saved to {output_file}")
