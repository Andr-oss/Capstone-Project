import os
import pandas as pd
import matplotlib.pyplot as plt

def plot_separate_coordinates(csv_file):
    # Check if the CSV file exists
    if not os.path.exists(csv_file):
        print(f"Error: File '{csv_file}' not found.")
        return

    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(csv_file)
    if data.empty:
        print("Error: CSV file is empty.")
        return

    # List the specific columns to plot (ignoring likelihood columns)
    columns_to_plot = [
        "Nose_x", "Nose_y",
        "Left_Ear_x", "Left_Ear_y",
        "Right_Ear_x", "Right_Ear_y",
        "Body_Center_x", "Body_Center_y",
        "Left_Body_x", "Left_Body_y",
        "Right_Body_x", "Right_Body_y",
        "Tail_Base_x", "Tail_Base_y"
    ]

    # Iterate over each specified column and create a separate graph
    for col in columns_to_plot:
        if col not in data.columns:
            print(f"Warning: Column '{col}' not found in CSV. Skipping.")
            continue

        # Create a new figure for the current column
        plt.figure()
        # Plot the data using the DataFrame's index as the x-axis
        plt.plot(data.index, data[col], marker='o', linestyle='-', markersize=3)
        # Add axis labels (no title to avoid clutter)
        plt.xlabel("Row Index")
        plt.ylabel(col)

    # Display all the figures
    plt.show()

if __name__ == '__main__':
    # Specify the full path to your CSV file
    csv_filename = "/Users/lionkk/Downloads/output22.csv"
    plot_separate_coordinates(csv_filename)