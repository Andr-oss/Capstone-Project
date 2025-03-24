import unittest
import numpy as np
import pandas as pd

def find_centroid(p1, p2, p3):
    """
    Calculate the centroid of three 2D points.
    Returns (NaN, NaN) if any coordinate is NaN.
    """
    if (pd.isna(p1[0]) or pd.isna(p1[1]) or
        pd.isna(p2[0]) or pd.isna(p2[1]) or
        pd.isna(p3[0]) or pd.isna(p3[1])):
        return np.nan, np.nan

    x = round((float(p1[0]) + float(p2[0]) + float(p3[0])) / 3, 2)
    y = round((float(p1[1]) + float(p2[1]) + float(p3[1])) / 3, 2)
    return x, y

class TestFindCentroid(unittest.TestCase):

    def test_centroid_with_valid_coordinates(self):
        """All points have valid (x,y)."""
        p1 = (1, 2)
        p2 = (3, 4)
        p3 = (5, 6)
        x, y = find_centroid(p1, p2, p3)
        # (1+3+5)/3 => 3.0, (2+4+6)/3 => 4.0
        self.assertEqual(x, 3.0)
        self.assertEqual(y, 4.0)

    def test_centroid_with_missing_coordinate(self):
        """At least one coordinate is NaN => should return (NaN, NaN)."""
        p1 = (np.nan, 2)
        p2 = (3, 4)
        p3 = (5, 6)
        x, y = find_centroid(p1, p2, p3)
        self.assertTrue(np.isnan(x))
        self.assertTrue(np.isnan(y))

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

class TestEstimateMissingPart(unittest.TestCase):
    def test_estimate_right_ear(self):
        """
        Right_Ear is missing => mirrored across Nose from Left_Ear:
        Right_Ear_x = 2*Nose_x - Left_Ear_x
        Right_Ear_y = 2*Nose_y - Left_Ear_y
        """
        row = {
            "Nose_x": 10, "Nose_y": 10,
            "Left_Ear_x": 8, "Left_Ear_y": 10,
            "Right_Ear_x": np.nan, "Right_Ear_y": np.nan
        }
        est_x, est_y = estimate_missing_part("Right_Ear", row)
        self.assertEqual(est_x, 12)  # 2*10 - 8 = 12
        self.assertEqual(est_y, 10)  # 2*10 - 10 = 10

    def test_estimate_left_ear(self):
        """
        Left_Ear is missing => mirrored across Nose from Right_Ear.
        """
        row = {
            "Nose_x": 10, "Nose_y": 10,
            "Left_Ear_x": np.nan, "Left_Ear_y": np.nan,
            "Right_Ear_x": 14, "Right_Ear_y": 8
        }
        est_x, est_y = estimate_missing_part("Left_Ear", row)
        self.assertEqual(est_x, 6)  # 2*10 - 14 = 6
        self.assertEqual(est_y, 12)  # 2*10 - 8  = 12

    def test_estimate_nose(self):
        """
        Nose is missing => average of the ears.
        """
        row = {
            "Nose_x": np.nan, "Nose_y": np.nan,
            "Left_Ear_x": 8, "Left_Ear_y": 10,
            "Right_Ear_x": 12, "Right_Ear_y": 14
        }
        est_x, est_y = estimate_missing_part("Nose", row)
        self.assertEqual(est_x, 10)  # (8 + 12) / 2
        self.assertEqual(est_y, 12)  # (10 + 14) / 2

    def test_estimate_missing_unavailable(self):
        """
        If needed references are also NaN => result should be (NaN, NaN).
        """
        row = {
            "Nose_x": np.nan, "Nose_y": np.nan,
            "Left_Ear_x": np.nan, "Left_Ear_y": np.nan,
            "Right_Ear_x": np.nan, "Right_Ear_y": np.nan,
            "Body_Center_x": np.nan, "Body_Center_y": np.nan
        }
        est_x, est_y = estimate_missing_part("Right_Ear", row)
        self.assertTrue(np.isnan(est_x))
        self.assertTrue(np.isnan(est_y))


if __name__ == '__main__':
    unittest.main()
