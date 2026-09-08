import os
import json
import hashlib

import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

DATASETS = {
    "IoMT-TrafficData": {
        "path": "data/raw/IoMT-TrafficData/output.csv",
        "label": "is_attack"
    },

    "MedSec-25": {
        "path": "data/raw/MedSec-25/MedSec-25.csv",
        "label": "Label"
    }
}

CHUNK_SIZE = 50000
TEST_SIZE = 0.20
RANDOM_SEED = 42


# ============================================================
# COLUMNS THAT SHOULD NOT BE USED AS FEATURES
# ============================================================

IDENTIFIER_COLUMNS = {
    "unnamed: 0",
    "flow id",
    "timestamp",
    "id.orig_h",
    "id.resp_h",
    "src ip",
    "dst ip"
}


# ============================================================
# DETERMINISTIC RANDOM VALUE
# ============================================================

def random_value(index, label):
    """
    Produces a deterministic number between 0 and 1.

    This allows us to create the same train/test split
    every time the program is run.
    """

    text = f"{label}_{index}_{RANDOM_SEED}"

    value = int(
        hashlib.md5(
            text.encode()
        ).hexdigest(),
        16
    )

    return (value % 1000000) / 1000000


# ============================================================
# DETERMINE FEATURE TYPES
# ============================================================

def determine_columns(path, label_column):

    print("\nInspecting columns...")

    sample = pd.read_csv(
        path,
        nrows=10000,
        low_memory=False
    )

    feature_columns = []
    numeric_columns = []
    categorical_columns = []
    removed_columns = []

    for column in sample.columns:

        column_name = column.lower()

        # Remove identifiers
        if column_name in IDENTIFIER_COLUMNS:

            removed_columns.append(column)
            continue

        # Remove label
        if column == label_column:

            removed_columns.append(column)
            continue

        feature_columns.append(column)

        # Try converting the column to numeric
        converted = pd.to_numeric(
            sample[column],
            errors="coerce"
        )

        numeric_ratio = converted.notna().mean()

        # If at least 80% of values are numeric,
        # treat the column as numeric.
        if numeric_ratio >= 0.80:

            numeric_columns.append(column)

        else:

            categorical_columns.append(column)

    print("\nTotal feature columns:", len(feature_columns))

    print("\nNumeric columns:", len(numeric_columns))

    print("\nCategorical columns:", len(categorical_columns))

    print("\nCategorical feature names:")

    for column in categorical_columns:
        print("  ", column)

    print("\nRemoved columns:")

    for column in removed_columns:
        print("  ", column)

    return (
        feature_columns,
        numeric_columns,
        categorical_columns,
        removed_columns
    )


# ============================================================
# FIND ALL CATEGORICAL VALUES
# ============================================================

def build_category_maps(
    path,
    categorical_columns
):

    category_maps = {
        column: {}
        for column in categorical_columns
    }

    if len(categorical_columns) == 0:
        return category_maps

    print("\nBuilding categorical mappings...")

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        usecols=categorical_columns,
        low_memory=False
    ):

        for column in categorical_columns:

            values = (
                chunk[column]
                .fillna("UNKNOWN")
                .astype(str)
                .unique()
            )

            for value in values:

                if value not in category_maps[column]:

                    category_maps[column][value] = (
                        len(category_maps[column])
                    )

    return category_maps


# ============================================================
# COUNT LABELS
# ============================================================

def count_labels(
    path,
    label_column
):

    print("\nCounting labels...")

    label_counts = {}

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        usecols=[label_column],
        low_memory=False
    ):

        counts = (
            chunk[label_column]
            .fillna("UNKNOWN")
            .astype(str)
            .value_counts()
        )

        for label, count in counts.items():

            label = str(label)

            label_counts[label] = (
                label_counts.get(label, 0)
                + int(count)
            )

    print("\nLabel distribution:")

    for label, count in label_counts.items():

        print(
            f"  {label}: {count:,}"
        )

    return label_counts


# ============================================================
# LABEL MAPPING
# ============================================================

def create_label_mapping(label_counts):

    labels = sorted(
        label_counts.keys()
    )

    mapping = {
        label: index
        for index, label in enumerate(labels)
    }

    print("\nLabel mapping:")

    for label, number in mapping.items():

        print(
            f"  {number} -> {label}"
        )

    return mapping


# ============================================================
# PROCESS A CHUNK
# ============================================================

def convert_chunk(
    chunk,
    feature_columns,
    numeric_columns,
    categorical_columns,
    category_maps
):

    X = chunk[
        feature_columns
    ].copy()

    # ----------------------------------------
    # Numeric columns
    # ----------------------------------------

    for column in numeric_columns:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce"
        )

    # ----------------------------------------
    # Categorical columns
    # ----------------------------------------

    for column in categorical_columns:

        X[column] = (
            X[column]
            .fillna("UNKNOWN")
            .astype(str)
            .map(category_maps[column])
            .fillna(-1)
        )

    # ----------------------------------------
    # Replace infinity
    # ----------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    return X


# ============================================================
# CREATE TRAIN / TEST SPLIT
# ============================================================

def is_training_row(
    row_index,
    label
):

    value = random_value(
        row_index,
        label
    )

    return value >= TEST_SIZE


# ============================================================
# COLLECT TRAINING MEDIANS
# ============================================================

def calculate_training_medians(
    path,
    label_column,
    feature_columns,
    numeric_columns,
    categorical_columns,
    category_maps
):

    print("\nCalculating training medians...")

    # We only need a reasonable sample to calculate medians.
    MEDIAN_SAMPLE_SIZE = 100000

    samples = []

    global_index = 0

    for chunk in pd.read_csv(
        path,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        labels = (
            chunk[label_column]
            .fillna("UNKNOWN")
            .astype(str)
            .values
        )

        X = convert_chunk(
            chunk,
            feature_columns,
            numeric_columns,
            categorical_columns,
            category_maps
        )

        train_indices = []

        for local_index, label in enumerate(labels):

            if is_training_row(
                global_index,
                label
            ):

                train_indices.append(
                    local_index
                )

            global_index += 1

            if len(samples) + len(train_indices) >= MEDIAN_SAMPLE_SIZE:
                break

        if train_indices:

            selected = X.iloc[
                train_indices
            ]

            samples.append(
                selected
            )

        if sum(
            len(sample)
            for sample in samples
        ) >= MEDIAN_SAMPLE_SIZE:

            break

    if samples:

        sample_data = pd.concat(
            samples,
            ignore_index=True
        )

        medians = sample_data.median(
            numeric_only=True
        )

    else:

        medians = pd.Series(
            0,
            index=feature_columns
        )

    medians = medians.reindex(
        feature_columns
    )

    medians = medians.fillna(0)

    return medians


# ============================================================
# CALCULATE TRAINING MINIMUM AND MAXIMUM
# ============================================================

def calculate_training_min_max(
    path,
    label_column,
    feature_columns,
    numeric_columns,
    categorical_columns,
    category_maps,
    training_medians
):

    print("\nCalculating training min/max values...")

    feature_min = pd.Series(
        np.inf,
        index=feature_columns,
        dtype=float
    )

    feature_max = pd.Series(
        -np.inf,
        index=feature_columns,
        dtype=float
    )

    global_index = 0

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):

        print(
            f"  Statistics chunk {chunk_number + 1}..."
        )

        labels = (
            chunk[label_column]
            .fillna("UNKNOWN")
            .astype(str)
            .values
        )

        X = convert_chunk(
            chunk,
            feature_columns,
            numeric_columns,
            categorical_columns,
            category_maps
        )

        train_indices = []

        for local_index, label in enumerate(labels):

            if is_training_row(
                global_index,
                label
            ):

                train_indices.append(
                    local_index
                )

            global_index += 1

        if not train_indices:
            continue

        X = X.iloc[
            train_indices
        ].copy()

        # Fill missing values using training medians
        X = X.fillna(
            training_medians
        )

        X = X.fillna(0)

        current_min = X.min()

        current_max = X.max()

        feature_min = pd.concat(
            [feature_min, current_min],
            axis=1
        ).min(axis=1)

        feature_max = pd.concat(
            [feature_max, current_max],
            axis=1
        ).max(axis=1)

    # Handle columns that somehow have no valid values
    feature_min = feature_min.replace(
        np.inf,
        0
    )

    feature_max = feature_max.replace(
        -np.inf,
        1
    )

    return feature_min, feature_max


# ============================================================
# NORMALIZE FEATURES
# ============================================================

def normalize(
    X,
    feature_min,
    feature_max
):

    denominator = (
        feature_max - feature_min
    )

    # Avoid division by zero for constant features
    denominator = denominator.replace(
        0,
        1
    )

    X = (
        X - feature_min
    ) / denominator

    # Protect against tiny numerical errors
    X = X.clip(
        lower=0,
        upper=1
    )

    return X


# ============================================================
# SAVE TRAIN AND TEST DATA
# ============================================================

def save_processed_dataset(
    path,
    label_column,
    feature_columns,
    numeric_columns,
    categorical_columns,
    category_maps,
    label_mapping,
    training_medians,
    feature_min,
    feature_max,
    output_folder
):

    print("\nCreating processed files...")

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    train_file = os.path.join(
        output_folder,
        "train.csv"
    )

    test_file = os.path.join(
        output_folder,
        "test.csv"
    )

    # Remove previous outputs
    if os.path.exists(train_file):
        os.remove(train_file)

    if os.path.exists(test_file):
        os.remove(test_file)

    global_index = 0

    train_rows = 0
    test_rows = 0

    first_train = True
    first_test = True

    for chunk_number, chunk in enumerate(
        pd.read_csv(
            path,
            chunksize=CHUNK_SIZE,
            low_memory=False
        )
    ):

        print(
            f"  Saving chunk {chunk_number + 1}..."
        )

        labels = (
            chunk[label_column]
            .fillna("UNKNOWN")
            .astype(str)
            .values
        )

        X = convert_chunk(
            chunk,
            feature_columns,
            numeric_columns,
            categorical_columns,
            category_maps
        )

        train_indices = []
        test_indices = []

        for local_index, label in enumerate(labels):

            if is_training_row(
                global_index,
                label
            ):

                train_indices.append(
                    local_index
                )

            else:

                test_indices.append(
                    local_index
                )

            global_index += 1

        # ====================================================
        # TRAINING DATA
        # ====================================================

        if train_indices:

            train_X = X.iloc[
                train_indices
            ].copy()

            train_labels = [
                label_mapping[labels[i]]
                for i in train_indices
            ]

            train_X = train_X.fillna(
                training_medians
            )

            train_X = train_X.fillna(0)

            train_X = normalize(
                train_X,
                feature_min,
                feature_max
            )

            train_X[label_column] = (
                train_labels
            )

            train_X.to_csv(
                train_file,
                mode="w" if first_train else "a",
                header=first_train,
                index=False
            )

            first_train = False

            train_rows += len(
                train_X
            )

        # ====================================================
        # TEST DATA
        # ====================================================

        if test_indices:

            test_X = X.iloc[
                test_indices
            ].copy()

            test_labels = [
                label_mapping[labels[i]]
                for i in test_indices
            ]

            test_X = test_X.fillna(
                training_medians
            )

            test_X = test_X.fillna(0)

            test_X = normalize(
                test_X,
                feature_min,
                feature_max
            )

            test_X[label_column] = (
                test_labels
            )

            test_X.to_csv(
                test_file,
                mode="w" if first_test else "a",
                header=first_test,
                index=False
            )

            first_test = False

            test_rows += len(
                test_X
            )

    print("\nFiles created:")

    print(
        f"  {train_file}"
    )

    print(
        f"  {test_file}"
    )

    print(
        f"\nTraining rows: {train_rows:,}"
    )

    print(
        f"Testing rows: {test_rows:,}"
    )

    return train_rows, test_rows


# ============================================================
# SAVE METADATA
# ============================================================

def save_metadata(
    output_folder,
    name,
    label_column,
    feature_columns,
    numeric_columns,
    categorical_columns,
    removed_columns,
    category_maps,
    label_mapping
):

    metadata = {
        "dataset": name,

        "label_column": label_column,

        "feature_count": len(
            feature_columns
        ),

        "feature_columns": feature_columns,

        "numeric_columns": numeric_columns,

        "categorical_columns": categorical_columns,

        "removed_columns": removed_columns,

        "label_mapping": label_mapping,

        "category_mappings": category_maps,

        "test_size": TEST_SIZE,

        "random_seed": RANDOM_SEED
    }

    metadata_file = os.path.join(
        output_folder,
        "metadata.json"
    )

    with open(
        metadata_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    print(
        f"\nMetadata saved to: {metadata_file}"
    )


# ============================================================
# PROCESS ONE DATASET
# ============================================================

def process_dataset(
    name,
    settings
):

    print("\n")
    print("=" * 70)
    print("PROCESSING:", name)
    print("=" * 70)

    path = settings["path"]

    label_column = settings["label"]

    # ----------------------------------------
    # 1. Determine feature types
    # ----------------------------------------

    (
        feature_columns,
        numeric_columns,
        categorical_columns,
        removed_columns
    ) = determine_columns(
        path,
        label_column
    )

    # ----------------------------------------
    # 2. Build categorical mappings
    # ----------------------------------------

    category_maps = build_category_maps(
        path,
        categorical_columns
    )

    # ----------------------------------------
    # 3. Count labels
    # ----------------------------------------

    label_counts = count_labels(
        path,
        label_column
    )

    # ----------------------------------------
    # 4. Create label mapping
    # ----------------------------------------

    label_mapping = create_label_mapping(
        label_counts
    )

    # ----------------------------------------
    # 5. Calculate training medians
    # ----------------------------------------

    training_medians = calculate_training_medians(
        path,
        label_column,
        feature_columns,
        numeric_columns,
        categorical_columns,
        category_maps
    )

    # ----------------------------------------
    # 6. Calculate training min/max
    # ----------------------------------------

    (
        feature_min,
        feature_max
    ) = calculate_training_min_max(
        path,
        label_column,
        feature_columns,
        numeric_columns,
        categorical_columns,
        category_maps,
        training_medians
    )

    # ----------------------------------------
    # 7. Output folder
    # ----------------------------------------

    output_folder = os.path.join(
        "data",
        "processed",
        name
    )

    # ----------------------------------------
    # 8. Save processed train/test
    # ----------------------------------------

    save_processed_dataset(
        path,
        label_column,
        feature_columns,
        numeric_columns,
        categorical_columns,
        category_maps,
        label_mapping,
        training_medians,
        feature_min,
        feature_max,
        output_folder
    )

    # ----------------------------------------
    # 9. Save metadata
    # ----------------------------------------

    save_metadata(
        output_folder,
        name,
        label_column,
        feature_columns,
        numeric_columns,
        categorical_columns,
        removed_columns,
        category_maps,
        label_mapping
    )

    print("\nFinished:", name)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for name, settings in DATASETS.items():

        process_dataset(
            name,
            settings
        )

    print("\n")
    print("=" * 70)
    print("ALL DATASETS PROCESSED SUCCESSFULLY!")
    print("=" * 70)