import pandas as pd
import numpy as np

def postprocess_csv(input_file, output_file):

    def estimate_missing_part(part, row):
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
