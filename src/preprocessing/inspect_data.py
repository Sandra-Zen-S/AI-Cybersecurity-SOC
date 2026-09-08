import pandas as pd


iomt_path = "data/raw/IoMT-TrafficData/output.csv"
medsec_path = "data/raw/MedSec-25/MedSec-25.csv"


def analyze_dataset(path, label_column, name):

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    label_counts = {}
    missing_counts = {}
    total_rows = 0

    # Read the dataset in chunks
    for chunk in pd.read_csv(path, chunksize=50000):

        total_rows += len(chunk)

        # Count labels
        counts = chunk[label_column].value_counts()

        for label, count in counts.items():
            label_counts[label] = label_counts.get(label, 0) + count

        # Count missing values
        missing = chunk.isnull().sum()

        for column, count in missing.items():
            missing_counts[column] = missing_counts.get(column, 0) + count

    print("\nTotal rows:")
    print(total_rows)

    print("\nLabel distribution:")
    for label, count in label_counts.items():
        print(label, ":", count)

    print("\nMissing values:")
    found_missing = False

    for column, count in missing_counts.items():
        if count > 0:
            print(column, ":", count)
            found_missing = True

    if not found_missing:
        print("No missing values")


# Analyze both datasets

analyze_dataset(
    iomt_path,
    "is_attack",
    "IoMT-TrafficData"
)

analyze_dataset(
    medsec_path,
    "Label",
    "MedSec-25"
)