import pandas as pd
from sklearn.impute import KNNImputer


def main():
    # Read the CSV file containing the rodent tracking data.
    # Expected columns: head_x, head_y, left_ear_x, left_ear_y, right_ear_x, right_ear_y,
    # body_center_x, body_center_y, left_body_x, left_body_y, right_body_x, right_body_y, tail_x, tail_y, etc.
    df = pd.read_csv("output.csv")

    # Create a KNN imputer with K=3
    imputer = KNNImputer(n_neighbors=3)

    # Perform the imputation. The imputer operates on the entire DataFrame assuming all columns are numeric.
    # If your CSV contains non-numeric columns, you may need to select only the relevant numeric columns.
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)

    # Save the imputed data to a new CSV file.
    df_imputed.to_csv("output_knn_filled.csv", index=False)
    print("KNN imputed CSV saved as output_knn_filled.csv")


if __name__ == "__main__":
    main()