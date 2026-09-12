import os
import json

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score


# ============================================================
# SETTINGS
# ============================================================

DATASETS = {
    "IoMT-TrafficData": {
        "train": "data/processed/IoMT-TrafficData/train.csv",
        "label": "is_attack"
    },

    "MedSec-25": {
        "train": "data/processed/MedSec-25/train.csv",
        "label": "Label"
    }
}

# Small sample because laptop has 8 GB RAM
SAMPLE_SIZE = 8000

# Lightweight WOA
POPULATION_SIZE = 5
MAX_ITERATIONS = 5

# Penalizes selecting too many features
FEATURE_PENALTY = 0.01

RANDOM_SEED = 42


# ============================================================
# LOAD SMALL REPRESENTATIVE SAMPLE
# ============================================================

def load_sample(path, label_column):

    print("\nReading a small sample...")

    # Read only 8,000 rows from the processed training file
    data = pd.read_csv(
        path,
        nrows=SAMPLE_SIZE
    )

    print("Sample shape:", data.shape)

    print("\nSample labels:")
    print(
        data[label_column].value_counts()
    )

    return data


# ============================================================
# PREPARE FEATURES
# ============================================================

def prepare_data(data, label_column):

    X = data.drop(
        columns=[label_column]
    )

    y = data[label_column]

    feature_names = X.columns.tolist()

    # Use float32 to reduce memory
    X = X.astype(
        np.float32
    )

    y = y.astype(
        int
    )

    return (
        X,
        y,
        feature_names
    )


# ============================================================
# FITNESS FUNCTION
# ============================================================

def evaluate_solution(
    position,
    X_train,
    X_valid,
    y_train,
    y_valid
):

    # Convert whale position into
    # selected / not selected
    selected = position >= 0.5

    # At least one feature must be selected
    if not np.any(selected):

        selected[
            np.argmax(position)
        ] = True

    selected_indices = np.where(
        selected
    )[0]

    # Select only chosen features
    X_train_selected = X_train[
        :, selected_indices
    ]

    X_valid_selected = X_valid[
        :, selected_indices
    ]

    # Small decision tree
    model = DecisionTreeClassifier(
        max_depth=4,
        random_state=RANDOM_SEED,
        class_weight="balanced"
    )

    model.fit(
        X_train_selected,
        y_train
    )

    predictions = model.predict(
        X_valid_selected
    )

    # Macro F1 works for binary and multiclass data
    score = f1_score(
        y_valid,
        predictions,
        average="macro",
        zero_division=0
    )

    # Feature-selection penalty
    feature_ratio = (
        len(selected_indices)
        / X_train.shape[1]
    )

    # Lower fitness is better
    fitness = (
        (1 - score)
        + FEATURE_PENALTY * feature_ratio
    )

    return (
        fitness,
        score,
        selected_indices
    )


# ============================================================
# WOA
# ============================================================

def whale_optimization(
    X_train,
    X_valid,
    y_train,
    y_valid,
    feature_names
):

    number_of_features = X_train.shape[1]

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    # --------------------------------------------------------
    # Create whale population
    # --------------------------------------------------------

    whales = rng.random(
        (
            POPULATION_SIZE,
            number_of_features
        )
    )

    best_position = None
    best_fitness = float("inf")
    best_f1 = 0
    best_features = None

    fitness_log = []

    # ========================================================
    # ITERATIONS
    # ========================================================

    for iteration in range(
        MAX_ITERATIONS
    ):

        print(
            f"\nWOA iteration "
            f"{iteration + 1}/{MAX_ITERATIONS}"
        )

        # ----------------------------------------------------
        # Evaluate whales
        # ----------------------------------------------------

        for i in range(
            POPULATION_SIZE
        ):

            fitness, score, selected_indices = (
                evaluate_solution(
                    whales[i],
                    X_train,
                    X_valid,
                    y_train,
                    y_valid
                )
            )

            if fitness < best_fitness:

                best_fitness = fitness

                best_position = (
                    whales[i].copy()
                )

                best_f1 = score

                best_features = (
                    selected_indices.copy()
                )

        print(
            "Best fitness:",
            round(best_fitness, 6)
        )

        print(
            "Best Macro F1:",
            round(best_f1, 6)
        )

        print(
            "Selected features:",
            len(best_features)
        )

        fitness_log.append(
            {
                "iteration": iteration + 1,
                "fitness": float(
                    best_fitness
                ),
                "macro_f1": float(
                    best_f1
                ),
                "selected_features": int(
                    len(best_features)
                )
            }
        )

        # ----------------------------------------------------
        # WOA parameter
        # ----------------------------------------------------

        a = 2 - (
            2 * iteration
            / MAX_ITERATIONS
        )

        # ----------------------------------------------------
        # Update whales
        # ----------------------------------------------------

        for i in range(
            POPULATION_SIZE
        ):

            r1 = rng.random(
                number_of_features
            )

            r2 = rng.random(
                number_of_features
            )

            A = (
                2 * a * r1
            ) - a

            C = (
                2 * r2
            )

            p = rng.random()

            l = rng.uniform(
                -1,
                1,
                number_of_features
            )

            # --------------------------------------------
            # Encircling / searching
            # --------------------------------------------

            if p < 0.5:

                if np.mean(
                    np.abs(A)
                ) < 1:

                    distance = np.abs(
                        C * best_position
                        - whales[i]
                    )

                    new_position = (
                        best_position
                        - A * distance
                    )

                else:

                    random_index = rng.integers(
                        0,
                        POPULATION_SIZE
                    )

                    random_whale = whales[
                        random_index
                    ]

                    distance = np.abs(
                        C * random_whale
                        - whales[i]
                    )

                    new_position = (
                        random_whale
                        - A * distance
                    )

            # --------------------------------------------
            # Spiral bubble-net attack
            # --------------------------------------------

            else:

                distance = np.abs(
                    best_position
                    - whales[i]
                )

                new_position = (
                    distance
                    * np.exp(l)
                    * np.cos(
                        2 * np.pi * l
                    )
                    + best_position
                )

            # --------------------------------------------
            # Convert to binary feature selection
            # --------------------------------------------

            probability = 1 / (
                1 + np.exp(
                    -np.clip(
                        new_position,
                        -50,
                        50
                    )
                )
            )

            random_values = rng.random(
                number_of_features
            )

            whales[i] = (
                probability
                > random_values
            ).astype(
                float
            )

    return (
        best_features,
        fitness_log
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    dataset_name,
    feature_names,
    selected_indices,
    fitness_log
):

    output_folder = os.path.join(
        "results",
        "woa",
        dataset_name
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Selected features
    # --------------------------------------------------------

    selected_features = []

    for index in selected_indices:

        selected_features.append(
            {
                "index": int(index),
                "name": feature_names[index]
            }
        )

    result = {
        "dataset": dataset_name,

        "total_features": len(
            feature_names
        ),

        "selected_feature_count": len(
            selected_features
        ),

        "selected_features": selected_features
    }

    feature_file = os.path.join(
        output_folder,
        "selected_features.json"
    )

    with open(
        feature_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Fitness log
    # --------------------------------------------------------

    fitness_file = os.path.join(
        output_folder,
        "fitness_log.csv"
    )

    pd.DataFrame(
        fitness_log
    ).to_csv(
        fitness_file,
        index=False
    )

    print(
        "\nSelected features saved to:"
    )

    print(feature_file)

    print(
        "\nFitness log saved to:"
    )

    print(fitness_file)


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name,
    settings
):

    print("\n")
    print("=" * 70)

    print(
        "WOA FEATURE SELECTION:",
        dataset_name
    )

    print("=" * 70)

    # --------------------------------------------------------
    # Load small sample
    # --------------------------------------------------------

    data = load_sample(
        settings["train"],
        settings["label"]
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    (
        X,
        y,
        feature_names
    ) = prepare_data(
        data,
        settings["label"]
    )

    print(
        "\nNumber of features:",
        len(feature_names)
    )

    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    X_train, X_valid, y_train, y_valid = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=RANDOM_SEED,
            stratify=y
        )
    )

    X_train = X_train.to_numpy(
        dtype=np.float32
    )

    X_valid = X_valid.to_numpy(
        dtype=np.float32
    )

    y_train = y_train.to_numpy()

    y_valid = y_valid.to_numpy()

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Validation samples:",
        len(X_valid)
    )

    # --------------------------------------------------------
    # Run WOA
    # --------------------------------------------------------

    (
        selected_indices,
        fitness_log
    ) = whale_optimization(
        X_train,
        X_valid,
        y_train,
        y_valid,
        feature_names
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    save_results(
        dataset_name,
        feature_names,
        selected_indices,
        fitness_log
    )

    print(
        "\nFinal selected feature count:",
        len(selected_indices)
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    for dataset_name, settings in DATASETS.items():

        process_dataset(
            dataset_name,
            settings
        )

    print("\n")
    print("=" * 70)
    print(
        "WOA FEATURE SELECTION COMPLETED!"
    )
    print("=" * 70)