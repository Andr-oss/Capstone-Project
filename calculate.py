import pandas as pd
import numpy as np

def circle_intersections(p1, r1, p2, r2):
    """
    Compute the intersections of two circles.
    p1, p2: centers (x, y)
    r1, r2: radii
    Returns two possible intersection points (if they exist) or None.
    """
    x1, y1 = p1
    x2, y2 = p2
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    # Check if circles intersect (or if one circle is contained in the other)
    if d > (r1 + r2) or d < abs(r1 - r2) or d == 0:
        return None
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    # Protect against floating point issues:
    h = np.sqrt(max(r1 ** 2 - a ** 2, 0))
    # Find midpoint between centers
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d
    # Two intersection points
    inter1 = (xm + h * (y1 - y2) / d, ym + h * (x2 - x1) / d)
    inter2 = (xm - h * (y1 - y2) / d, ym - h * (x2 - x1) / d)
    return inter1, inter2


def choose_head_solution(sol1, sol2, left_ear, right_ear):
    """
    From the two intersection solutions, choose the one that fits our assumption:
    In image coordinates (origin at top-left, y increasing downwards),
    the head is assumed to be above (i.e. with a smaller y-value) than both ears.
    """
    if sol1 is None or sol2 is None:
        return None
    y1 = sol1[1]
    y2 = sol2[1]
    ear_y1 = left_ear[1]
    ear_y2 = right_ear[1]
    candidates = []
    if y1 < ear_y1 and y1 < ear_y2:
        candidates.append(sol1)
    if y2 < ear_y1 and y2 < ear_y2:
        candidates.append(sol2)
    if candidates:
        # choose the candidate with the smallest y (i.e. highest on image)
        return min(candidates, key=lambda pt: pt[1])
    return sol1  # fallback


def main():
    # Read the CSV file (ensure that the file has appropriate header names)
    df = pd.read_csv("output.csv")

    # Expected columns (modify if needed):
    # "head_x", "head_y", "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y",
    # "body_center_x", "body_center_y", "left_body_x", "left_body_y", "right_body_x", "right_body_y",
    # "tail_x", "tail_y"

    # 1. Compute calibration values from rows with complete head–ears data.
    valid_head = df.dropna(subset=["head_x", "head_y", "left_ear_x", "left_ear_y", "right_ear_x", "right_ear_y"]).copy()
    if len(valid_head) > 0:
        valid_head["dist_head_left"] = np.sqrt((valid_head["head_x"] - valid_head["left_ear_x"]) ** 2 +
                                               (valid_head["head_y"] - valid_head["left_ear_y"]) ** 2)
        valid_head["dist_head_right"] = np.sqrt((valid_head["head_x"] - valid_head["right_ear_x"]) ** 2 +
                                                (valid_head["head_y"] - valid_head["right_ear_y"]) ** 2)
        avg_head_left = valid_head["dist_head_left"].mean()
        avg_head_right = valid_head["dist_head_right"].mean()
    else:
        avg_head_left = None
        avg_head_right = None

    # 2. Compute average tail offset from body center (if available)
    valid_tail = df.dropna(subset=["body_center_x", "body_center_y", "tail_x", "tail_y"]).copy()
    if len(valid_tail) > 0:
        valid_tail["tail_offset_x"] = valid_tail["tail_x"] - valid_tail["body_center_x"]
        valid_tail["tail_offset_y"] = valid_tail["tail_y"] - valid_tail["body_center_y"]
        avg_tail_offset_x = valid_tail["tail_offset_x"].mean()
        avg_tail_offset_y = valid_tail["tail_offset_y"].mean()
    else:
        avg_tail_offset_x = 0
        avg_tail_offset_y = 0

    # Process each row to fill missing values
    for idx, row in df.iterrows():
        # Determine if any of the head or ear values are missing.
        head_missing = pd.isna(row["head_x"]) or pd.isna(row["head_y"])
        left_missing = pd.isna(row["left_ear_x"]) or pd.isna(row["left_ear_y"])
        right_missing = pd.isna(row["right_ear_x"]) or pd.isna(row["right_ear_y"])

        # ----- Head–Ears Triangle -----
        # If head is missing but both ears are present, use circle intersections.
        if head_missing and (not left_missing) and (not right_missing) \
                and avg_head_left is not None and avg_head_right is not None:
            left_ear = (row["left_ear_x"], row["left_ear_y"])
            right_ear = (row["right_ear_x"], row["right_ear_y"])
            intersections = circle_intersections(left_ear, avg_head_left, right_ear, avg_head_right)
            if intersections is not None:
                head_point = choose_head_solution(intersections[0], intersections[1], left_ear, right_ear)
                df.at[idx, "head_x"] = head_point[0]
                df.at[idx, "head_y"] = head_point[1]

        # If an ear is missing (and head and the other ear are present), assume symmetry:
        # For left ear missing, estimate it as the reflection of the right ear across the head.
        if left_missing and (not head_missing) and (not right_missing):
            head_point = np.array([row["head_x"], row["head_y"]])
            right_ear = np.array([row["right_ear_x"], row["right_ear_y"]])
            left_ear = 2 * head_point - right_ear
            df.at[idx, "left_ear_x"] = left_ear[0]
            df.at[idx, "left_ear_y"] = left_ear[1]

        # Similarly for right ear missing.
        if right_missing and (not head_missing) and (not left_missing):
            head_point = np.array([row["head_x"], row["head_y"]])
            left_ear = np.array([row["left_ear_x"], row["left_ear_y"]])
            right_ear = 2 * head_point - left_ear
            df.at[idx, "right_ear_x"] = right_ear[0]
            df.at[idx, "right_ear_y"] = right_ear[1]

        # ----- Body and Tail Points -----
        # For body points, we assume left body, center body, and right body are collinear.
        body_center_missing = pd.isna(row["body_center_x"]) or pd.isna(row["body_center_y"])
        left_body_missing = pd.isna(row["left_body_x"]) or pd.isna(row["left_body_y"])
        right_body_missing = pd.isna(row["right_body_x"]) or pd.isna(row["right_body_y"])

        # If the center is missing but both sides are available, fill it as the midpoint.
        if body_center_missing and (not left_body_missing) and (not right_body_missing):
            left_body = np.array([row["left_body_x"], row["left_body_y"]])
            right_body = np.array([row["right_body_x"], row["right_body_y"]])
            center_body = (left_body + right_body) / 2
            df.at[idx, "body_center_x"] = center_body[0]
            df.at[idx, "body_center_y"] = center_body[1]

        # If a side point is missing, estimate it by symmetry about the center.
        if left_body_missing and (not body_center_missing) and (not right_body_missing):
            center_body = np.array([row["body_center_x"], row["body_center_y"]])
            right_body = np.array([row["right_body_x"], row["right_body_y"]])
            left_body = 2 * center_body - right_body
            df.at[idx, "left_body_x"] = left_body[0]
            df.at[idx, "left_body_y"] = left_body[1]

        if right_body_missing and (not body_center_missing) and (not left_body_missing):
            center_body = np.array([row["body_center_x"], row["body_center_y"]])
            left_body = np.array([row["left_body_x"], row["left_body_y"]])
            right_body = 2 * center_body - left_body
            df.at[idx, "right_body_x"] = right_body[0]
            df.at[idx, "right_body_y"] = right_body[1]

        # For tail: if missing and body center is available, use the average tail offset.
        tail_missing = pd.isna(row["tail_x"]) or pd.isna(row["tail_y"])
        if tail_missing and not body_center_missing:
            df.at[idx, "tail_x"] = row["body_center_x"] + avg_tail_offset_x
            df.at[idx, "tail_y"] = row["body_center_y"] + avg_tail_offset_y

    # Save the updated DataFrame to a new CSV file.
    df.to_csv("output_filled.csv", index=False)
    print("Filled CSV saved as output_filled.csv")


if __name__ == "__main__":
    main()
