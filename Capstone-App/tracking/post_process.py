import pandas as pd
import numpy as np

def postprocess_csv(input_file, output_file):
    """
    Process the file to prepare for visualization.
    Parameters:
        input_file (str): Path to the input csv file.
        output_file (str): Path to the output csv file.
    """

    def is_manual_reset(df, frame_idx, threshold=50, min_parts_jumping=4):
        """
        Check if a frame likely represents a manual reset based on multiple body parts
        showing large jumps in either x or y directions.
        """
        jump_count = 0
        body_parts = ["Nose", "Left_Ear", "Right_Ear", "Body_Center", "Left_Body", "Right_Body", "Tail_Base"]

        for part in body_parts:
            x_col, y_col = f"{part}_x", f"{part}_y"
            if frame_idx == 0 or x_col not in df.columns or y_col not in df.columns:
                continue

            prev_x, curr_x = df.at[frame_idx - 1, x_col], df.at[frame_idx, x_col]
            prev_y, curr_y = df.at[frame_idx - 1, y_col], df.at[frame_idx, y_col]

            if pd.notna(prev_x) and pd.notna(curr_x):
                if abs(curr_x - prev_x) > threshold:
                    jump_count += 1
                    continue  # already counted, skip y check

            if pd.notna(prev_y) and pd.notna(curr_y):
                if abs(curr_y - prev_y) > threshold:
                    jump_count += 1

        return jump_count >= min_parts_jumping

    def remove_outliers(df, part, threshold=100):
        """
        Remove outliers from the coordinate columns of a given body part by setting
        values with large frame-to-frame jumps to NaN.
        Parameters:
            df (pd.DataFrame): The dataframe with coordinate data.
            part (str): The name of the body part.
            threshold (float): Distance threshold beyond which values are considered outliers.
        """
        x_col, y_col = f"{part}_x", f"{part}_y"
        if x_col not in df.columns or y_col not in df.columns:
            return
        x_diff = df[x_col].diff().abs()
        y_diff = df[y_col].diff().abs()
        outlier_mask = (x_diff > threshold) | (y_diff > threshold)
        df.loc[outlier_mask, [x_col, y_col]] = np.nan

    def estimate_missing_part(part, row):
        """
        Estimate the missing part if missing in file.
        Parameters:
            part (str): Part missing.
            row (pd.Series): Row of the file.
        """
        if part == 'Right_Ear' and not (pd.isna(row["Nose_x"]) or pd.isna(row["Left_Ear_x"])):
            return 2 * row["Nose_x"] - row["Left_Ear_x"], 2 * row["Nose_y"] - row["Left_Ear_y"]
        elif part == 'Left_Ear' and not (pd.isna(row["Nose_x"]) or pd.isna(row["Right_Ear_x"])):
            return 2 * row["Nose_x"] - row["Right_Ear_x"], 2 * row["Nose_y"] - row["Right_Ear_y"]
        elif part == 'Nose' and not (pd.isna(row["Left_Ear_x"]) or pd.isna(row["Right_Ear_x"])):
            return (row["Left_Ear_x"] + row["Right_Ear_x"]) / 2, (row["Left_Ear_y"] + row["Right_Ear_y"]) / 2
        elif part == 'Right_Body' and not (pd.isna(row["Body_Center_x"]) or pd.isna(row["Left_Body_x"])):
            return 2 * row["Body_Center_x"] - row["Left_Body_x"], 2 * row["Body_Center_y"] - row["Left_Body_y"]
        elif part == 'Left_Body' and not (pd.isna(row["Body_Center_x"]) or pd.isna(row["Right_Body_x"])):
            return 2 * row["Body_Center_x"] - row["Right_Body_x"], 2 * row["Body_Center_y"] - row["Right_Body_y"]
        elif part == 'Tail_Base' and not pd.isna(row["Body_Center_x"]):
            return row["Body_Center_x"], row["Body_Center_y"] + 10
        return np.nan, np.nan

    df = pd.read_csv(input_file, header=[1, 2])
    df.columns = [f"{col1}_{col2}" if pd.notna(col2) else col1 for col1, col2 in df.columns]
    df.rename(columns={df.columns[0]: "Frame"}, inplace=True)

    def find_centroid(p1, p2, p3):
        """
        Find the centroid of three points.
        Parameters:
            p1 (pd.Series): First point.
            p2 (pd.Series): Second point.
            p3 (pd.Series): Third point.
        """
        if (pd.isna(p1[0]) or pd.isna(p1[1]) or
                pd.isna(p2[0]) or pd.isna(p2[1]) or
                pd.isna(p3[0]) or pd.isna(p3[1])):
            return np.nan, np.nan
        x = round((float(p1[0]) + float(p2[0]) + float(p3[0])) / 3, 2)
        y = round((float(p1[1]) + float(p2[1]) + float(p3[1])) / 3, 2)
        return x, y

    body_parts = ["Nose", "Left_Ear", "Right_Ear", "Body_Center", "Left_Body", "Right_Body", "Tail_Base"]
    for part in body_parts:
        for suffix in ['_x', '_y']:
            col = f"{part}{suffix}"
            if col in df.columns:
                df[col] = pd.to_numeric(df[col].replace('', np.nan), errors='coerce')

    for part in body_parts:
        remove_outliers(df, part)
        for suffix in ['_x', '_y']:
            col = f"{part}{suffix}"
            if col in df.columns:
                df[col] = df[col].interpolate(method='linear', limit=3)

    for idx, row in df.iterrows():
        for part in body_parts:
            x_col = f"{part}_x"
            y_col = f"{part}_y"
            if x_col in df.columns and y_col in df.columns:
                if pd.isna(row[x_col]) or pd.isna(row[y_col]):
                    est_x, est_y = estimate_missing_part(part, row)
                    if not (pd.isna(est_x) or pd.isna(est_y)):
                        df.at[idx, x_col] = est_x
                        df.at[idx, y_col] = est_y

    df["Head_Center_x"] = np.nan
    df["Head_Center_y"] = np.nan
    for idx, row in df.iterrows():
        if not (pd.isna(row["Nose_x"]) or pd.isna(row["Left_Ear_x"]) or pd.isna(row["Right_Ear_x"])):
            x, y = find_centroid((row["Nose_x"], row["Nose_y"]),
                                 (row["Left_Ear_x"], row["Left_Ear_y"]),
                                 (row["Right_Ear_x"], row["Right_Ear_y"]))
            df.at[idx, "Head_Center_x"] = x
            df.at[idx, "Head_Center_y"] = y

    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].round(2)
    df.to_csv(output_file, index=False)
    print(f"✅ Post-processing done. Saved to: {output_file}")
