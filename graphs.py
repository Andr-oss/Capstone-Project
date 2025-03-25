import os
import pandas as pd
import matplotlib.pyplot as plt


def extract_landmarks(columns):
    """
    Given a list of column names, find all pairs that represent landmarks.
    A valid pair is one where there is a column ending with '_x' and a corresponding
    column with the same prefix ending with '_y'.
    Returns a list of landmark names (the prefix part).
    """
    landmarks = []
    for col in columns:
        if col.endswith("_x"):
            landmark = col[:-2]  # Remove the "_x" suffix to get the landmark name.
            if f"{landmark}_y" in columns:
                landmarks.append(landmark)
    return landmarks


def plot_landmarks(csv_file):
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)
    if data.empty:
        print("Error: CSV file is empty.")
        return

    # Dynamically extract landmark names from the CSV header
    landmarks = extract_landmarks(data.columns)
    if not landmarks:
        print("Error: No valid landmark columns found in the CSV file.")
        return
    print(f"Identified landmarks: {landmarks}")

    # Increase figure size to accommodate more elements
    plt.figure(figsize=(10, 8))

    # Process each row in the CSV file
    for index, row in data.iterrows():
        x_coords = []
        y_coords = []
        labels = []

        # For each detected landmark, extract the x and y values
        for lm in landmarks:
            x_col = f"{lm}_x"
            y_col = f"{lm}_y"
            try:
                x_val = row[x_col]
                y_val = row[y_col]
                x_coords.append(x_val)
                y_coords.append(y_val)
                labels.append(lm)
            except KeyError:
                print(f"Warning: Missing columns for landmark '{lm}' in row {index + 1}. Skipping.")

        # Plot the points for the current row
        plt.scatter(x_coords, y_coords, label=f"Row {index + 1}")

        # Annotate each point with its landmark name
        for lm, x, y in zip(labels, x_coords, y_coords):
            plt.annotate(lm, (x, y), textcoords="offset points", xytext=(5, 5), ha='center')

    # Option 1: Use a fixed legend location to improve performance
    plt.legend(loc="upper right")

    # Option 2: If too many legend entries, consider commenting out the legend
    # plt.legend().set_visible(False)

    plt.xlabel("X coordinate")
    plt.ylabel("Y coordinate")
    plt.title("Landmark Points from CSV Data")

    # Manually adjust layout if tight_layout() doesn't work well
    plt.subplots_adjust(top=0.9, bottom=0.1, left=0.1, right=0.9)

    # Display the plot
    plt.show()


if __name__ == '__main__':
    # Provide the full path to your CSV file
    csv_filename = "/Users/lionkk/Downloads/output22.csv"
    plot_landmarks(csv_filename)