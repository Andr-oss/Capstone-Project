import pandas as pd

h5_path = "D:/Capstone-Project/AndrewFirstTraining-Andrew-2025-03-08/training-datasets/iteration-0/UnaugmentedDataSet_Capstone-project-DLCMar8/CollectedData_Andrew.h5"

with pd.HDFStore(h5_path, "r") as store:
    df = store["/keypoints"]
    print(f"Total labeled frames: {len(df)}")
    print(f"Columns (Keypoints): {df.columns}")
