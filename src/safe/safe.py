import sys
import io
import os
import json
import random
import warnings

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import roc_auc_score
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore")


# ============================================================
# M3 - SAFE
# Self-supervised Masked Autoencoder
# ============================================================


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)

PROCESSED_DIR = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed"
)

WOA_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "woa"
)

SAFE_DIR = os.path.join(
    PROJECT_ROOT,
    "results",
    "safe"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

# Read CSV files in small chunks
CHUNK_SIZE = 20000

# Maximum normal samples used for training
# (will use all available if fewer exist)
MAX_NORMAL_TRAIN_SAMPLES = 50000

# Maximum test samples used for evaluation
MAX_TEST_SAMPLES = 20000

# Fraction of normal training data held out for
# validation-based threshold tuning
VALIDATION_FRACTION = 0.20

# Small sample of attack traffic used ONLY for
# threshold tuning on the validation set
MAX_ATTACK_VALIDATION_SAMPLES = 2000

# Neural network
BATCH_SIZE = 128
EPOCHS = 30
LEARNING_RATE = 0.0005

# Percentage of features hidden during self-supervised training
MASK_RATIO = 0.20

# Fallback: if validation-based threshold fails,
# use this percentile on normal reconstruction errors
THRESHOLD_PERCENTILE = 95

# Architecture / loss defaults (used when no tuned config exists)
LATENT_DIM = 32
DROPOUT = 0.1
LOSS = "huber"            # "huber" (SmoothL1) or "mse"

# Anomaly scorer on top of the trained autoencoder:
#   "recon"      - mean squared reconstruction error
#   "latent_lof" - Local Outlier Factor on the encoder's latent features
#                  (the novelty-detector design of the original SAFE paper)
#   "hybrid"     - LOF on [latent features, log reconstruction error]
#   "auto"       - pick the best of the above on VALIDATION ROC-AUC
SCORER = "auto"
LOF_NEIGHBORS = [10, 20, 50]     # candidates tried by "auto"
LOF_MAX_FIT = 20000              # benign flows used to fit LOF
SCORER_TOLERANCE = 0.002         # simpler scorer kept if within this AUC

# Upper bound on the false-positive rate on benign VALIDATION flows when
# choosing the threshold (a SOC cannot act on mostly-false alarms).
MAX_VALIDATION_FPR = 0.05

# Validation ROC-AUC above which the score distributions count as "well
# separated" (Youden's J); 0.76 = Phi(1/sqrt(2)), i.e. Cohen's d = 1.
SEPARATION_AUC = 0.76

# Columns that encode the label itself and must never be features.
# In IoMT-TrafficData, 'traffic' is the attack-type name label-encoded.
LEAKY_COLUMNS = {"traffic", "label", "Label", "is_attack", "attack_type", "Attack Type"}

TUNING_DIR = os.path.join(PROJECT_ROOT, "results", "tuning")

# Final per-dataset hyperparameters, chosen on VALIDATION ROC-AUC in the
# tune_safe.py --quick study (1 seed, epochs evaluated 1..40).
#
# IoMT-TrafficData: all settings reach val ROC-AUC ~0.999 (saturated), so the
#   defaults are kept. 30 epochs: early stopping on validation loss stops at
#   epoch 29; more epochs add training time without improving detection.
# MedSec-25: a LIST of mask ratios means "train one model per ratio and keep
#   the one whose selected scorer has the best VALIDATION ROC-AUC".
#   (With reconstruction scoring, mask 0.0 was best: 0.766 vs 0.705; with
#   encoder+LOF scoring the comparison ablation suggested masking helps, so
#   the ratio is re-selected on validation instead of fixed.)
# Earlier quick study, reconstruction scoring: mask_ratio 0.0 beats 0.2 on validation (0.766 vs 0.705) and on
#   test (0.695 vs 0.655). Epoch study selected 38 epochs (early stop at 39).
#   lr 0.003 was NOT adopted: +0.01 on validation but -0.06 on test (unstable).
FINAL_CONFIGS = {
    "IoMT-TrafficData": {
        "epochs": 30,
        "mask_ratio": 0.2,
        "lr": 5e-4,
        "batch_size": 128,
        "latent": 32,
        "dropout": 0.1,
        "loss": "huber",
        "scorer": "auto"
    },
    "MedSec-25": {
        "epochs": 38,
        "mask_ratio": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7],   # chosen on validation together with the scorer
        "lr": 5e-4,
        "batch_size": 128,
        "latent": 32,
        "dropout": 0.1,
        "loss": "huber",
        "scorer": "auto"
    }
}


# ============================================================
# PER-DATASET HYPERPARAMETERS (from tune_safe.py)
# ============================================================

def load_safe_config(dataset_name):
    """
    Default hyperparameters, overridden by the validation-selected
    configuration in results/tuning/<dataset>/best_config.json
    (written by src/evaluation/tune_safe.py). Configs from --quick
    tuning runs (single seed) are ignored.
    """

    config = {
        "epochs": EPOCHS,
        "mask_ratio": MASK_RATIO,
        "lr": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "latent": LATENT_DIM,
        "dropout": DROPOUT,
        "loss": LOSS,
        "scorer": SCORER,
        "source": "safe.py defaults"
    }

    if dataset_name in FINAL_CONFIGS:

        config.update(FINAL_CONFIGS[dataset_name])
        config["source"] = "FINAL_CONFIGS in safe.py (quick tuning study)"

    path = os.path.join(TUNING_DIR, dataset_name, "best_config.json")

    if os.path.exists(path):

        with open(path, "r", encoding="utf-8") as file:
            saved = json.load(file)

        is_quick = saved.get("quick", len(saved.get("seeds", [])) < 2)

        if is_quick:
            print(
                f"\nNOTE: {path} comes from a --quick tuning run; "
                "ignoring it and using safe.py defaults."
            )
        else:
            tuned = dict(saved["tuned_config"])
            tuned.pop("scorer", None)   # scorer is chosen on validation by select_scorer
            config.update(tuned)
            config["source"] = path

    return config


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cpu")


# ============================================================
# RANDOM SEEDS
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


# ============================================================
# DIRECTORY CREATION
# ============================================================

def ensure_directory(path):

    os.makedirs(
        path,
        exist_ok=True
    )


# ============================================================
# DETECT LABEL COLUMN
# ============================================================

def detect_label_column(csv_path):
    """
    Automatically detect the label column.

    Possible names from the original datasets / preprocessing:
        label
        Label
        is_attack
        target
        Target
    """

    header = pd.read_csv(
        csv_path,
        nrows=0
    )

    columns = list(
        header.columns
    )

    possible_names = [
        "label",
        "Label",
        "is_attack",
        "target",
        "Target"
    ]

    # First try known names
    for name in possible_names:

        if name in columns:

            return name

    # Fallback: case-insensitive search
    lower_map = {
        str(column).lower(): column
        for column in columns
    }

    for name in possible_names:

        if name.lower() in lower_map:

            return lower_map[
                name.lower()
            ]

    raise ValueError(
        "\nCould not automatically detect "
        "the label column.\n\n"
        "Available columns:\n"
        + "\n".join(
            str(column)
            for column in columns
        )
    )


# ============================================================
# DETERMINE NORMAL LABEL
# ============================================================

def determine_normal_label(
    dataset_name,
    train_path,
    label_column
):
    """
    Determine which numeric label represents normal traffic.

    M1 in this project maps normal/benign traffic to 0.

    We verify this from the data instead of blindly assuming
    that an attack label is 1.
    """

    # --------------------------------------------------------
    # Try metadata first
    # --------------------------------------------------------

    metadata_path = os.path.join(
        os.path.dirname(train_path),
        "metadata.json"
    )

    if os.path.exists(
        metadata_path
    ):

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8"
            ) as file:

                metadata = json.load(
                    file
                )

            label_mapping = metadata.get(
                "label_mapping",
                {}
            )

            if isinstance(
                label_mapping,
                dict
            ):

                # Look for obvious normal names
                for original_label, numeric_label in label_mapping.items():

                    name = str(
                        original_label
                    ).strip().lower()

                    if name in {
                        "normal",
                        "benign",
                        "0"
                    }:

                        try:

                            return int(
                                numeric_label
                            )

                        except (
                            TypeError,
                            ValueError
                        ):

                            pass

        except Exception:
            pass

    # --------------------------------------------------------
    # Project's M1 normal-label convention
    # --------------------------------------------------------

    return 0


# ============================================================
# LOAD WOA SELECTED FEATURES
# ============================================================

def load_selected_features(
    dataset_name
):
    """
    Load features produced by M2 WOA.

    Actual project format:

    {
        "dataset": "IoMT-TrafficData",
        "total_features": 98,
        "selected_feature_count": 45,
        "selected_features": [
            {
                "index": 0,
                "name": "id.orig_p"
            },
            ...
        ]
    }
    """

    json_path = os.path.join(
        WOA_DIR,
        dataset_name,
        "selected_features.json"
    )

    if not os.path.exists(
        json_path
    ):

        raise FileNotFoundError(
            "\nWOA result not found:\n"
            f"{json_path}\n\n"
            "Run M2 WOA before M3 SAFE."
        )

    with open(
        json_path,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(
            file
        )

    # --------------------------------------------------------
    # Dictionary format
    # --------------------------------------------------------

    if isinstance(
        data,
        dict
    ):

        if "selected_features" not in data:

            raise ValueError(
                "\nWOA JSON does not contain "
                "'selected_features'."
            )

        selected_features = data[
            "selected_features"
        ]

    # --------------------------------------------------------
    # Direct list format
    # --------------------------------------------------------

    elif isinstance(
        data,
        list
    ):

        selected_features = data

    else:

        raise ValueError(
            "\nUnsupported WOA JSON format."
        )

    if not selected_features:

        raise ValueError(
            "\nWOA returned zero selected features."
        )

    return selected_features


# ============================================================
# RESOLVE WOA FEATURES
# ============================================================

def resolve_feature_columns(
    train_path,
    selected_features
):
    """
    Convert WOA feature objects into actual CSV columns.

    Supports:

    1. {"index": 0, "name": "duration"}
    2. "duration"
    3. 0
    """

    header = pd.read_csv(
        train_path,
        nrows=0
    )

    all_columns = list(
        header.columns
    )

    # --------------------------------------------------------
    # CASE 1
    # WOA dictionary objects
    # --------------------------------------------------------

    if all(
        isinstance(
            item,
            dict
        )
        for item in selected_features
    ):

        selected_columns = []

        for item in selected_features:

            if "name" not in item:

                raise ValueError(
                    "\nA WOA feature object "
                    "does not contain 'name'."
                )

            feature_name = str(
                item["name"]
            )

            if feature_name not in all_columns:

                raise ValueError(
                    "\nWOA feature does not exist "
                    "in processed CSV:\n"
                    f"{feature_name}"
                )

            selected_columns.append(
                feature_name
            )

        return selected_columns

    # --------------------------------------------------------
    # CASE 2
    # WOA feature names
    # --------------------------------------------------------

    if all(
        isinstance(
            item,
            str
        )
        for item in selected_features
    ):

        selected_columns = []

        for feature_name in selected_features:

            if feature_name not in all_columns:

                raise ValueError(
                    "\nWOA feature does not exist "
                    "in processed CSV:\n"
                    f"{feature_name}"
                )

            selected_columns.append(
                feature_name
            )

        return selected_columns

    # --------------------------------------------------------
    # CASE 3
    # WOA feature indices
    # --------------------------------------------------------

    if all(
        isinstance(
            item,
            (int, np.integer)
        )
        for item in selected_features
    ):

        label_names = {
            "label",
            "Label",
            "is_attack",
            "target",
            "Target"
        }

        feature_columns = [
            column
            for column in all_columns
            if column not in label_names
        ]

        selected_columns = []

        for index in selected_features:

            index = int(
                index
            )

            if (
                index < 0
                or
                index >= len(feature_columns)
            ):

                raise ValueError(
                    f"\nWOA feature index {index} "
                    "is outside the feature range."
                )

            selected_columns.append(
                feature_columns[index]
            )

        return selected_columns

    # --------------------------------------------------------
    # Unsupported
    # --------------------------------------------------------

    raise ValueError(
        "\nUnsupported WOA selected-feature format."
    )


# ============================================================
# LOAD NORMAL TRAINING DATA
# ============================================================

def load_normal_training_data(
    train_path,
    selected_columns,
    label_column,
    normal_label,
    max_samples
):
    """
    Load only normal traffic from the training dataset.

    The full dataset is NEVER loaded into RAM.

    Collects all normal samples (up to max_samples)
    and randomly shuffles them for unbiased training.
    """

    print(
        "\nLoading normal training samples..."
    )

    selected_columns = list(
        selected_columns
    )

    usecols = (
        selected_columns
        + [label_column]
    )

    normal_parts = []

    total_normal = 0

    # --------------------------------------------------------
    # Chunked reading
    # --------------------------------------------------------

    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        normal_chunk = chunk[
            chunk[label_column] == normal_label
        ]

        if len(
            normal_chunk
        ) == 0:

            continue

        normal_parts.append(
            normal_chunk[
                selected_columns
            ]
        )

        total_normal += len(
            normal_chunk
        )

        if total_normal >= max_samples:

            break

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if not normal_parts:

        raise ValueError(
            "\nNo normal training samples found.\n"
            f"Label column: {label_column}\n"
            f"Expected normal label: {normal_label}"
        )

    # --------------------------------------------------------
    # Combine and shuffle
    # --------------------------------------------------------

    X_normal = pd.concat(
        normal_parts,
        ignore_index=True
    )

    X_normal = X_normal.iloc[
        :max_samples
    ]

    # Shuffle to avoid ordering bias
    X_normal = X_normal.sample(
        frac=1.0,
        random_state=RANDOM_SEED
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Numeric conversion
    # --------------------------------------------------------

    X_normal = X_normal.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # Replace invalid values
    X_normal = X_normal.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Fill missing values
    X_normal = X_normal.fillna(
        0.0
    )

    # Float32 reduces RAM usage
    X_normal = X_normal.astype(
        np.float32
    )

    print(
        "Normal training samples loaded: "
        f"{len(X_normal):,}"
    )

    return X_normal.values


# ============================================================
# LOAD ATTACK VALIDATION DATA
# ============================================================

def load_attack_validation_data(
    train_path,
    selected_columns,
    label_column,
    normal_label,
    max_samples
):
    """
    Load a small sample of ATTACK traffic from the
    TRAINING set for validation-based threshold tuning.

    This is NOT data leakage because:
    - We use TRAINING data, not test data
    - Labels from training data are used only for
      threshold selection, not for model training
    """

    print(
        "\nLoading attack validation samples "
        "from training set..."
    )

    selected_columns = list(
        selected_columns
    )

    usecols = (
        selected_columns
        + [label_column]
    )

    # Uniform random sample over ALL attack rows in the file
    # (keeps the attack-class mix representative; the previous
    # version took the first rows only, which in a class-ordered
    # CSV means one or two attack types).
    rng = np.random.default_rng(RANDOM_SEED)

    kept = None

    for chunk in pd.read_csv(
        train_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        attack_chunk = chunk[
            chunk[label_column] != normal_label
        ][selected_columns].copy()

        if len(attack_chunk) == 0:

            continue

        attack_chunk["_r"] = rng.random(
            len(attack_chunk)
        )

        kept = (
            attack_chunk if kept is None
            else pd.concat([kept, attack_chunk], ignore_index=True)
        ).nsmallest(max_samples, "_r")

    # --------------------------------------------------------
    # Verify
    # --------------------------------------------------------

    if kept is None or len(kept) == 0:

        print(
            "WARNING: No attack validation "
            "samples found in training data."
        )

        return None

    X_attack = kept.drop(
        columns=["_r"]
    ).reset_index(
        drop=True
    )

    X_attack = X_attack.apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_attack = X_attack.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_attack = X_attack.fillna(
        0.0
    )

    X_attack = X_attack.astype(
        np.float32
    )

    print(
        "Attack validation samples loaded: "
        f"{len(X_attack):,}"
    )

    return X_attack.values


# ============================================================
# LOAD TEST DATA — STRATIFIED SAMPLING
# ============================================================

def load_test_data(
    test_path,
    selected_columns,
    label_column,
    normal_label,
    max_samples,
    return_index=False
):
    """
    Load test data with STRATIFIED sampling.

    Instead of taking the first N rows (which creates
    bias if the CSV is ordered by class), this function:

    1. First pass: counts total rows per class
    2. Second pass: loads a proportional sample per class

    This ensures the test set distribution matches the
    true distribution, giving representative accuracy.
    """

    print(
        "\nLoading test data (stratified)..."
    )

    selected_columns = list(
        selected_columns
    )

    usecols = (
        selected_columns
        + [label_column]
    )

    # --------------------------------------------------------
    # PASS 1: Count rows per label
    # --------------------------------------------------------

    label_counts = {}
    total_rows = 0

    for chunk in pd.read_csv(
        test_path,
        usecols=[label_column],
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        vc = chunk[label_column].value_counts()

        for label_val, count in vc.items():

            label_val = int(label_val)

            label_counts[label_val] = (
                label_counts.get(label_val, 0)
                + count
            )

        total_rows += len(chunk)

    print(
        f"Total test rows: {total_rows:,}"
    )

    print(
        "Label distribution:"
    )

    for label_val in sorted(label_counts.keys()):

        pct = (
            label_counts[label_val]
            / total_rows
            * 100
        )

        is_normal = (
            "Normal" if label_val == normal_label
            else "Attack"
        )

        print(
            f"  Label {label_val} ({is_normal}): "
            f"{label_counts[label_val]:,} "
            f"({pct:.2f}%)"
        )

    # --------------------------------------------------------
    # Calculate per-class sample budget
    # --------------------------------------------------------

    actual_max = min(
        max_samples,
        total_rows
    )

    # Proportional allocation per class
    samples_per_label = {}

    for label_val in label_counts:

        fraction = (
            label_counts[label_val]
            / total_rows
        )

        samples_per_label[label_val] = max(
            1,
            int(
                round(fraction * actual_max)
            )
        )

    # --------------------------------------------------------
    # PASS 2: Reservoir sampling per class
    # --------------------------------------------------------

    # Collect indices for reservoir sampling
    collected = {
        label_val: []
        for label_val in label_counts
    }

    row_offset = 0

    rng = np.random.RandomState(
        RANDOM_SEED
    )

    for chunk in pd.read_csv(
        test_path,
        usecols=usecols,
        chunksize=CHUNK_SIZE,
        low_memory=False
    ):

        labels = pd.to_numeric(
            chunk[label_column],
            errors="coerce"
        ).fillna(0).astype(int)

        for label_val in label_counts:

            mask = (labels == label_val)

            label_rows = chunk[mask]

            budget = samples_per_label[label_val]

            for _, row in label_rows.iterrows():

                if len(
                    collected[label_val]
                ) < budget:

                    collected[label_val].append(
                        row
                    )

                else:

                    # Reservoir sampling
                    j = rng.randint(
                        0,
                        row_offset + 1
                    )

                    if j < budget:

                        collected[label_val][j] = row

            row_offset += mask.sum()

    # --------------------------------------------------------
    # Combine all classes
    # --------------------------------------------------------

    all_rows = []

    for label_val in sorted(collected.keys()):

        all_rows.extend(
            collected[label_val]
        )

    test_df = pd.DataFrame(all_rows)

    # Shuffle to mix classes
    test_df = test_df.sample(
        frac=1.0,
        random_state=RANDOM_SEED
    )

    # Row number of each sampled flow in test.csv (0-based, data rows).
    # M4 uses it to look up the real flow record for each prediction.
    test_row_index = test_df.index.to_numpy()

    test_df = test_df.reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    X_test_df = test_df[
        selected_columns
    ].apply(
        pd.to_numeric,
        errors="coerce"
    )

    X_test_df = X_test_df.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X_test_df = X_test_df.fillna(
        0.0
    )

    X_test = X_test_df.astype(
        np.float32
    ).values

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    y_test = pd.to_numeric(
        test_df[label_column],
        errors="coerce"
    ).fillna(
        0
    ).astype(
        int
    ).values

    print(
        f"Stratified test samples loaded: "
        f"{len(X_test):,}"
    )

    # Show what we actually sampled
    unique, counts = np.unique(
        y_test,
        return_counts=True
    )

    for u, c in zip(unique, counts):

        is_normal = (
            "Normal" if u == normal_label
            else "Attack"
        )

        print(
            f"  Sampled label {u} ({is_normal}): "
            f"{c:,}"
        )

    if return_index:

        return X_test, y_test, test_row_index

    return X_test, y_test


# ============================================================
# PER-FEATURE EXPLANATION
# ============================================================

def per_feature_errors(
    model,
    X
):
    """Squared reconstruction error per feature, shape (n_samples, n_features)."""

    model.eval()

    out = []

    with torch.no_grad():

        for i in range(0, len(X), 4096):

            batch = torch.tensor(
                X[i:i + 4096],
                dtype=torch.float32
            )

            out.append(
                ((model(batch) - batch) ** 2).numpy()
            )

    return np.concatenate(out)


def explain_predictions(
    model,
    X_test,
    X_normal_val,
    feature_names,
    top_k=5
):
    """
    For every test flow, the top-k features whose reconstruction error is
    largest relative to the average error on normal validation traffic.
    Returned as JSON strings: [[feature, error_vs_normal], ...].
    """

    baseline = per_feature_errors(
        model,
        X_normal_val
    ).mean(axis=0) + 1e-9

    ratios = per_feature_errors(
        model,
        X_test
    ) / baseline

    top = np.argsort(-ratios, axis=1)[:, :top_k]

    return [
        json.dumps([
            [feature_names[j], round(float(ratios[i, j]), 2)]
            for j in top[i]
        ])
        for i in range(len(X_test))
    ]


# ============================================================
# SAFE MASKED AUTOENCODER — IMPROVED
# ============================================================

class MaskedAutoencoder(
    nn.Module
):
    """
    Improved self-supervised masked autoencoder.

    Encoder:
        Input
          ↓
        128 + BatchNorm + ReLU + Dropout
          ↓
        64 + BatchNorm + ReLU + Dropout
          ↓
        32 latent representation

    Decoder:
        32
          ↓
        64 + BatchNorm + ReLU + Dropout
          ↓
        128 + BatchNorm + ReLU
          ↓
        Output (linear — no Sigmoid)

    Key changes:
    - Wider layers for more capacity
    - BatchNorm for training stability
    - Dropout for regularization
    - Linear output instead of Sigmoid to avoid
      gradient saturation when many features are
      near 0 or near 1
    """

    def __init__(
        self,
        input_dim,
        latent_dim=32,
        dropout=0.1
    ):

        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(
                input_dim,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                128,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                latent_dim
            ),

            nn.ReLU()
        )

        self.decoder = nn.Sequential(

            nn.Linear(
                latent_dim,
                64
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.Dropout(dropout),

            nn.Linear(
                64,
                128
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.Linear(
                128,
                input_dim
            )

            # No Sigmoid — linear output
            # Data is already normalized to [0,1]
            # by M1. Sigmoid would create gradient
            # saturation for features near 0 or 1.
        )

    def forward(
        self,
        x
    ):

        latent = self.encoder(
            x
        )

        reconstructed = self.decoder(
            latent
        )

        return reconstructed


# ============================================================
# TRAIN SAFE
# ============================================================

def train_safe_model(
    X_train,
    input_dim,
    config
):
    """
    Train SAFE using normal traffic only.

    Key change: loss is computed on ALL features,
    not just masked ones. The mask is applied as input
    corruption (denoising autoencoder), but the model
    must reconstruct the full input. This aligns the
    training objective with evaluation.
    """

    print(
        "\nCreating SAFE model..."
    )

    model = MaskedAutoencoder(
        input_dim,
        latent_dim=config["latent"],
        dropout=config["dropout"]
    )

    model = model.to(
        DEVICE
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["lr"],
        weight_decay=1e-5
    )

    # Huber loss (SmoothL1) is more robust
    # than MSE for features with occasional
    # outlier values
    loss_function = (
        nn.MSELoss() if config["loss"] == "mse"
        else nn.SmoothL1Loss()
    )

    X_tensor = torch.tensor(
        X_train,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        drop_last=True
    )

    print(
        f"\nTraining SAFE for "
        f"{config['epochs']} epochs "
        f"(mask ratio {config['mask_ratio']}, lr {config['lr']}, "
        f"batch {config['batch_size']}, loss {config['loss']})..."
    )

    print(
        f"Architecture: {input_dim} -> 128 -> "
        f"64 -> {config['latent']} -> 64 -> 128 -> {input_dim}"
    )

    print(
        f"Training samples: {len(X_train):,}"
    )

    model.train()

    epochs = config["epochs"]

    for epoch in range(
        1,
        epochs + 1
    ):

        total_loss = 0.0

        batch_count = 0

        for (batch,) in loader:

            batch = batch.to(
                DEVICE
            )

            # ------------------------------------------------
            # Create random feature mask
            # ------------------------------------------------

            mask = (
                torch.rand_like(batch)
                < config["mask_ratio"]
            )

            # ------------------------------------------------
            # Mask features (input corruption)
            # ------------------------------------------------

            masked_batch = batch.clone()

            masked_batch[
                mask
            ] = 0.0

            # ------------------------------------------------
            # Reconstruction
            # ------------------------------------------------

            reconstructed = model(
                masked_batch
            )

            # ------------------------------------------------
            # Loss on ALL features
            # (not just masked ones)
            # This aligns training with evaluation
            # ------------------------------------------------

            loss = loss_function(
                reconstructed,
                batch
            )

            # ------------------------------------------------
            # Backpropagation
            # ------------------------------------------------

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
            )

            batch_count += 1

        average_loss = (
            total_loss
            /
            max(
                batch_count,
                1
            )
        )

        if epoch % 5 == 0 or epoch == 1:

            print(
                f"Epoch "
                f"[{epoch:02d}/{epochs}] "
                f"Loss: "
                f"{average_loss:.6f}"
            )

    return model


# ============================================================
# RECONSTRUCTION ERROR
# ============================================================

def reconstruction_errors(
    model,
    X
):
    """
    Reconstruction error is used as SAFE anomaly score.

    Uses MSE per sample (mean of squared errors across
    all features for each sample).
    """

    model.eval()

    X_tensor = torch.tensor(
        X,
        dtype=torch.float32
    )

    dataset = TensorDataset(
        X_tensor
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    errors = []

    with torch.no_grad():

        for (batch,) in loader:

            batch = batch.to(
                DEVICE
            )

            reconstructed = model(
                batch
            )

            mse = torch.mean(
                (
                    reconstructed
                    - batch
                ) ** 2,
                dim=1
            )

            errors.extend(
                mse.cpu().numpy()
            )

    return np.asarray(
        errors,
        dtype=np.float32
    )


# ============================================================
# THRESHOLD — VALIDATION-BASED (Youden's J)
# ============================================================

# ============================================================
# ANOMALY SCORERS ON TOP OF THE AUTOENCODER
# ============================================================

def latent_features(
    model,
    X
):
    """Encoder output (latent representation) for every row of X."""

    model.eval()

    out = []

    with torch.no_grad():

        for i in range(0, len(X), 4096):

            batch = torch.tensor(
                X[i:i + 4096],
                dtype=torch.float32
            )

            out.append(
                model.encoder(batch).numpy()
            )

    return np.concatenate(out)


class SAFEScorer:
    """
    Turns a trained autoencoder into an anomaly scorer.
    Higher score = more anomalous.
    """

    def __init__(
        self,
        model,
        kind="recon",
        n_neighbors=20
    ):

        self.model = model
        self.kind = kind
        self.n_neighbors = n_neighbors
        self.lof = None
        self.scaler = None

    def _lof_input(
        self,
        X
    ):

        z = latent_features(
            self.model,
            X
        )

        if self.kind == "hybrid":

            err = np.log10(
                reconstruction_errors(self.model, X) + 1e-9
            ).reshape(-1, 1)

            z = np.hstack([z, err])

        return z

    def fit(
        self,
        X_train,
        seed=RANDOM_SEED
    ):

        if self.kind == "recon":

            return self

        if len(X_train) > LOF_MAX_FIT:

            idx = np.random.default_rng(seed).choice(
                len(X_train),
                LOF_MAX_FIT,
                replace=False
            )

            X_train = X_train[idx]

        Z = self._lof_input(X_train)

        self.scaler = StandardScaler().fit(Z)

        self.lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            novelty=True,
            n_jobs=-1
        ).fit(
            self.scaler.transform(Z)
        )

        return self

    def score(
        self,
        X
    ):

        if self.kind == "recon":

            return reconstruction_errors(
                self.model,
                X
            )

        Z = self.scaler.transform(
            self._lof_input(X)
        )

        # score_samples is the negative LOF; negate so higher = more anomalous
        return (-self.lof.score_samples(Z)).astype(np.float32)

    @property
    def name(self):

        if self.kind == "recon":

            return "recon"

        return f"{self.kind}(k={self.n_neighbors})"


def select_scorer(
    model,
    X_train,
    X_val_normal,
    X_val_attack,
    requested="auto"
):
    """
    Choose the anomaly scorer on VALIDATION ROC-AUC only.
    A simpler scorer (recon < latent_lof < hybrid) is kept unless a more
    complex one improves validation ROC-AUC by more than SCORER_TOLERANCE.
    """

    if requested != "auto":

        chosen = SAFEScorer(model, requested, LOF_NEIGHBORS[1]).fit(X_train)

        table = []

        if X_val_attack is not None and len(X_val_attack) > 0:

            s = np.r_[chosen.score(X_val_normal), chosen.score(X_val_attack)]
            y = np.r_[np.zeros(len(X_val_normal)), np.ones(len(X_val_attack))]
            table = [{"scorer": chosen.name, "val_roc_auc": round(float(roc_auc_score(y, s)), 4)}]

        return chosen, table

    if X_val_attack is None or len(X_val_attack) == 0:

        print("\nNo attack validation data: using reconstruction scorer.")

        return SAFEScorer(model, "recon"), []

    candidates = [SAFEScorer(model, "recon")]

    for kind in ("latent_lof", "hybrid"):

        for k in LOF_NEIGHBORS:

            candidates.append(SAFEScorer(model, kind, k))

    y_val = np.r_[
        np.zeros(len(X_val_normal)),
        np.ones(len(X_val_attack))
    ]

    print("\nSelecting anomaly scorer on validation ROC-AUC...")

    table = []

    for c in candidates:

        c.fit(X_train)

        s = np.r_[c.score(X_val_normal), c.score(X_val_attack)]

        auc = float(roc_auc_score(y_val, s))

        table.append({"scorer": c.name, "val_roc_auc": round(auc, 4)})

        print(f"  {c.name:<18} validation ROC-AUC {auc:.4f}")

    best_auc = max(r["val_roc_auc"] for r in table)

    # candidates are ordered simplest first
    for c, r in zip(candidates, table):

        if r["val_roc_auc"] >= best_auc - SCORER_TOLERANCE:

            print(f"Selected scorer: {c.name}")

            return c, table


def calculate_threshold_validation(
    normal_errors,
    attack_errors
):
    """
    Find the optimal threshold using Youden's J statistic
    on the validation set.

    Youden's J = Sensitivity + Specificity - 1

    This maximizes the overall classification performance
    (balances both normal and attack detection).

    No data leakage: uses training-set normal validation
    split and training-set attack samples only.
    """

    print(
        "\nCalculating optimal threshold "
        "using validation set..."
    )

    # --------------------------------------------------------
    # Candidate thresholds
    # --------------------------------------------------------

    all_errors = np.concatenate(
        [normal_errors, attack_errors]
    )

    # Candidates from all errors + fine-grained candidates around normal validation error distribution
    candidates_all = np.percentile(
        all_errors,
        np.arange(0.1, 100, 0.25)
    )

    normal_candidates = np.percentile(
        normal_errors,
        np.linspace(80, 99.9, 300)
    )

    candidates = np.unique(
        np.sort(
            np.concatenate(
                [candidates_all, normal_candidates]
            )
        )
    )

    # --------------------------------------------------------
    # Labels: 0 = normal, 1 = attack
    # --------------------------------------------------------

    y_val = np.concatenate([
        np.zeros(len(normal_errors)),
        np.ones(len(attack_errors))
    ])

    scores_val = np.concatenate(
        [normal_errors, attack_errors]
    )

    # --------------------------------------------------------
    # Calculate Cohen's d separation on validation set
    # --------------------------------------------------------

    mean_normal = np.mean(normal_errors)
    mean_attack = np.mean(attack_errors)
    var_normal = np.var(normal_errors)
    var_attack = np.var(attack_errors)
    pooled_std = np.sqrt(0.5 * (var_normal + var_attack)) + 1e-9
    cohen_d = float((mean_attack - mean_normal) / pooled_std)

    # --------------------------------------------------------
    # Find optimal thresholds for Youden J and Max F1
    # --------------------------------------------------------

    # Lowest threshold that keeps validation FPR within the cap; used as the
    # fallback and as the lower bound for all candidates.
    fpr_floor = float(
        np.percentile(normal_errors, 100.0 * (1.0 - MAX_VALIDATION_FPR))
    )

    candidates = candidates[candidates >= fpr_floor]

    if len(candidates) == 0:
        candidates = np.array([fpr_floor])

    best_j = -1.0
    thresh_youden = fpr_floor

    best_f1 = -1.0
    thresh_f1 = thresh_youden

    for threshold in candidates:

        predictions = (
            scores_val > threshold
        ).astype(int)

        tp = np.sum(
            (predictions == 1) & (y_val == 1)
        )

        tn = np.sum(
            (predictions == 0) & (y_val == 0)
        )

        fp = np.sum(
            (predictions == 1) & (y_val == 0)
        )

        fn = np.sum(
            (predictions == 0) & (y_val == 1)
        )

        sensitivity = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        precision = tp / max(tp + fp, 1e-9)
        recall = tp / max(tp + fn, 1e-9)
        f1 = (2 * precision * recall) / max(precision + recall, 1e-9)
        j = sensitivity + specificity - 1.0

        if j > best_j:
            best_j = j
            thresh_youden = float(threshold)

        if f1 > best_f1:
            best_f1 = f1
            thresh_f1 = float(threshold)

    # --------------------------------------------------------
    # Adaptive threshold selection
    # --------------------------------------------------------

    # Separation is measured with the validation ROC-AUC, which is scale-free
    # (Cohen's d breaks down on heavy-tailed scores such as LOF). Under the
    # equal-variance binormal model AUC = Phi(d / sqrt(2)), so AUC >= 0.76
    # corresponds to the original rule d >= 1.
    from scipy.stats import norm
    val_auc = float(roc_auc_score(y_val, scores_val))
    d_equiv = float(np.sqrt(2.0) * norm.ppf(min(max(val_auc, 1e-6), 1 - 1e-6)))

    if val_auc >= SEPARATION_AUC:
        best_threshold = thresh_youden
        strategy = (f"Youden's J (validation AUC = {val_auc:.3f} >= {SEPARATION_AUC}, "
                    f"equivalent d = {d_equiv:.2f})")
    else:
        best_threshold = thresh_f1
        strategy = (f"Validation Max F1 (validation AUC = {val_auc:.3f} < {SEPARATION_AUC}, "
                    f"equivalent d = {d_equiv:.2f})")

    print(
        f"Threshold selection strategy: {strategy}, "
        f"validation FPR capped at {MAX_VALIDATION_FPR:.0%}"
    )

    print(
        f"Validation Youden's J: {best_j:.4f} | Max F1: {best_f1:.4f}"
    )

    print(
        f"Selected threshold: {best_threshold:.6f}"
    )

    return best_threshold


def calculate_threshold_percentile(
    normal_errors
):
    """
    Fallback: threshold = percentile of
    normal reconstruction errors.
    """

    threshold = float(
        np.percentile(
            normal_errors,
            THRESHOLD_PERCENTILE
        )
    )

    return threshold


# ============================================================
# PRINT SCORE DISTRIBUTION STATISTICS
# ============================================================

def print_score_statistics(
    normal_errors,
    attack_errors
):
    """
    Print detailed reconstruction error statistics
    to understand separation quality.
    """

    print(
        "\n" + "=" * 60
    )

    print(
        "Anomaly Score Statistics"
    )

    print(
        "=" * 60
    )

    print(
        "\nNormal traffic anomaly scores:"
    )

    print(
        f"  Count:   {len(normal_errors):,}"
    )

    print(
        f"  Min:     {np.min(normal_errors):.6f}"
    )

    print(
        f"  Max:     {np.max(normal_errors):.6f}"
    )

    print(
        f"  Mean:    {np.mean(normal_errors):.6f}"
    )

    print(
        f"  Median:  {np.median(normal_errors):.6f}"
    )

    print(
        f"  Std:     {np.std(normal_errors):.6f}"
    )

    print(
        f"  P95:     "
        f"{np.percentile(normal_errors, 95):.6f}"
    )

    print(
        f"  P99:     "
        f"{np.percentile(normal_errors, 99):.6f}"
    )

    print(
        "\nAttack traffic anomaly scores:"
    )

    print(
        f"  Count:   {len(attack_errors):,}"
    )

    print(
        f"  Min:     {np.min(attack_errors):.6f}"
    )

    print(
        f"  Max:     {np.max(attack_errors):.6f}"
    )

    print(
        f"  Mean:    {np.mean(attack_errors):.6f}"
    )

    print(
        f"  Median:  {np.median(attack_errors):.6f}"
    )

    print(
        f"  Std:     {np.std(attack_errors):.6f}"
    )

    # --------------------------------------------------------
    # Separation assessment
    # --------------------------------------------------------

    normal_mean = np.mean(normal_errors)
    attack_mean = np.mean(attack_errors)
    normal_std = np.std(normal_errors)
    attack_std = np.std(attack_errors)

    pooled_std = np.sqrt(
        (normal_std ** 2 + attack_std ** 2) / 2
    )

    if pooled_std > 0:

        separation = abs(
            attack_mean - normal_mean
        ) / pooled_std

    else:

        separation = 0.0

    print(
        f"\nSeparation (Cohen's d): "
        f"{separation:.4f}"
    )

    if separation > 2.0:

        print(
            "  → Excellent separation"
        )

    elif separation > 1.0:

        print(
            "  → Good separation"
        )

    elif separation > 0.5:

        print(
            "  → Moderate separation"
        )

    else:

        print(
            "  → Weak separation — "
            "model may struggle to "
            "distinguish normal from attack"
        )


# ============================================================
# EVALUATE SAFE
# ============================================================

def evaluate_safe(
    model,
    X_test,
    y_test,
    normal_label,
    threshold
):
    """
    Convert reconstruction scores into:

        0 = Normal
        1 = Anomaly

    For multiclass datasets such as MedSec-25,
    every label other than the normal label is treated
    as an anomaly.
    """

    print(
        "\nCalculating anomaly scores..."
    )

    scores = (
        model.score(X_test) if isinstance(model, SAFEScorer)
        else reconstruction_errors(model, X_test)
    )

    predictions = (
        scores > threshold
    ).astype(
        int
    )

    # --------------------------------------------------------
    # Convert multiclass ground truth to binary
    # --------------------------------------------------------

    y_true_binary = (
        y_test != normal_label
    ).astype(
        int
    )

    normal_predictions = int(
        np.sum(
            predictions == 0
        )
    )

    anomaly_predictions = int(
        np.sum(
            predictions == 1
        )
    )

    actual_normal = int(
        np.sum(
            y_true_binary == 0
        )
    )

    actual_anomaly = int(
        np.sum(
            y_true_binary == 1
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    true_negative = int(
        np.sum(
            (predictions == 0)
            &
            (y_true_binary == 0)
        )
    )

    false_positive = int(
        np.sum(
            (predictions == 1)
            &
            (y_true_binary == 0)
        )
    )

    false_negative = int(
        np.sum(
            (predictions == 0)
            &
            (y_true_binary == 1)
        )
    )

    true_positive = int(
        np.sum(
            (predictions == 1)
            &
            (y_true_binary == 1)
        )
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = (
        true_positive
        + true_negative
    ) / max(
        len(y_true_binary),
        1
    )

    precision = (
        true_positive
        /
        max(
            true_positive
            + false_positive,
            1
        )
    )

    recall = (
        true_positive
        /
        max(
            true_positive
            + false_negative,
            1
        )
    )

    if (
        precision + recall
    ) > 0:

        f1 = (
            2
            * precision
            * recall
            /
            (
                precision
                + recall
            )
        )

    else:

        f1 = 0.0

    specificity = (
        true_negative
        /
        max(
            true_negative
            + false_positive,
            1
        )
    )

    # --------------------------------------------------------
    # ROC-AUC
    # --------------------------------------------------------

    roc_auc = compute_roc_auc(
        y_true_binary,
        scores
    )

    # --------------------------------------------------------
    # PR-AUC
    # --------------------------------------------------------

    pr_auc = compute_pr_auc(
        y_true_binary,
        scores
    )

    # --------------------------------------------------------
    # Score statistics (test set)
    # --------------------------------------------------------

    normal_mask = (y_true_binary == 0)
    attack_mask = (y_true_binary == 1)

    if np.sum(normal_mask) > 0 and np.sum(attack_mask) > 0:

        print_score_statistics(
            scores[normal_mask],
            scores[attack_mask]
        )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 60
    )

    print(
        "SAFE Detection Results"
    )

    print(
        "=" * 60
    )

    print(
        f"Threshold: "
        f"{threshold:.6f}"
    )

    print(
        f"\nTest samples: "
        f"{len(y_true_binary):,}"
    )

    print(
        f"Actual normal samples: "
        f"{actual_normal:,}"
    )

    print(
        f"Actual anomaly samples: "
        f"{actual_anomaly:,}"
    )

    print(
        f"Predicted normal samples: "
        f"{normal_predictions:,}"
    )

    print(
        f"Predicted anomaly samples: "
        f"{anomaly_predictions:,}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        "----------------"
    )

    print(
        f"True Negative (TN):  {true_negative}"
    )

    print(
        f"False Positive (FP): {false_positive}"
    )

    print(
        f"False Negative (FN): {false_negative}"
    )

    print(
        f"True Positive (TP):  {true_positive}"
    )

    print(
        "\nPerformance Metrics"
    )

    print(
        "-------------------"
    )

    print(
        f"Accuracy:    {accuracy:.4f} "
        f"({accuracy * 100:.2f}%)"
    )

    print(
        f"Precision:   {precision:.4f} "
        f"({precision * 100:.2f}%)"
    )

    print(
        f"Recall:      {recall:.4f} "
        f"({recall * 100:.2f}%)"
    )

    print(
        f"F1-score:    {f1:.4f} "
        f"({f1 * 100:.2f}%)"
    )

    print(
        f"Specificity: {specificity:.4f} "
        f"({specificity * 100:.2f}%)"
    )

    print(
        f"ROC-AUC:     {roc_auc:.4f}"
    )

    print(
        f"PR-AUC:      {pr_auc:.4f}"
    )

    extra = imbalance_aware_metrics(
        true_positive, false_positive, true_negative, false_negative
    )

    print(
        "\nFlag rate per original class "
        "(normal class = false-positive rate; others = detection rate):"
    )

    for label_value in np.unique(y_test):

        mask = (y_test == label_value)

        kind = "normal" if label_value == normal_label else "attack"

        print(
            f"  Label {label_value} ({kind}): "
            f"{predictions[mask].mean():.4f} of {int(mask.sum()):,} flows flagged"
        )

    print(
        f"FPR:         {extra['false_positive_rate']:.4f}"
    )

    print(
        f"Balanced accuracy: {extra['balanced_accuracy']:.4f}"
    )

    print(
        f"MCC:         {extra['mcc']:.4f}"
    )

    print(
        f"\nTrivial baseline (flag every flow as attack): "
        f"accuracy {extra['baseline_all_attack_accuracy']:.4f}, "
        f"F1 {extra['baseline_all_attack_f1']:.4f}"
    )

    return (
        scores,
        predictions,
        y_true_binary,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    )


# ============================================================
# IMBALANCE-AWARE METRICS
# ============================================================

def imbalance_aware_metrics(
    tp,
    fp,
    tn,
    fn
):
    tpr = tp / max(tp + fn, 1)
    tnr = tn / max(tn + fp, 1)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0
    pos_share = (tp + fn) / max(tp + fp + tn + fn, 1)

    return {
        "specificity": float(tnr),
        "false_positive_rate": float(1.0 - tnr),
        "balanced_accuracy": float((tpr + tnr) / 2.0),
        "mcc": float(mcc),
        "baseline_all_attack_accuracy": float(pos_share),
        "baseline_all_attack_f1": float(2 * pos_share / (1 + pos_share)) if pos_share > 0 else 0.0,
    }


# ============================================================
# ROC-AUC (computed without sklearn)
# ============================================================

def compute_roc_auc(
    y_true,
    scores
):
    """
    Compute ROC-AUC using the trapezoidal rule.

    Implemented manually to avoid requiring sklearn.
    """

    try:

        # Sort by descending score
        desc_indices = np.argsort(
            scores
        )[::-1]

        y_sorted = y_true[
            desc_indices
        ]

        n_pos = np.sum(y_true == 1)
        n_neg = np.sum(y_true == 0)

        if n_pos == 0 or n_neg == 0:

            return 0.0

        tp = 0
        fp = 0
        auc = 0.0
        tp_prev = 0
        fp_prev = 0

        scores_sorted = scores[
            desc_indices
        ]

        prev_score = None

        for i in range(len(y_sorted)):

            if (
                prev_score is not None
                and
                scores_sorted[i] != prev_score
            ):

                # Trapezoidal rule
                auc += (
                    (fp - fp_prev)
                    * (tp + tp_prev)
                    / 2.0
                )

                tp_prev = tp
                fp_prev = fp

            if y_sorted[i] == 1:

                tp += 1

            else:

                fp += 1

            prev_score = scores_sorted[i]

        # Final trapezoid
        auc += (
            (fp - fp_prev)
            * (tp + tp_prev)
            / 2.0
        )

        auc = auc / (n_pos * n_neg)

        return float(auc)

    except Exception:

        return 0.0


# ============================================================
# PR-AUC (computed without sklearn)
# ============================================================

def compute_pr_auc(
    y_true,
    scores
):
    """
    Compute PR-AUC using the trapezoidal rule.
    """

    try:

        desc_indices = np.argsort(
            scores
        )[::-1]

        y_sorted = y_true[
            desc_indices
        ]

        n_pos = np.sum(y_true == 1)

        if n_pos == 0:

            return 0.0

        tp = 0
        fp = 0
        precisions = [1.0]
        recalls = [0.0]

        scores_sorted = scores[
            desc_indices
        ]

        prev_score = None

        for i in range(len(y_sorted)):

            if y_sorted[i] == 1:

                tp += 1

            else:

                fp += 1

            if (
                prev_score is None
                or
                scores_sorted[i] != prev_score
            ):

                p = tp / max(tp + fp, 1)
                r = tp / n_pos

                precisions.append(p)
                recalls.append(r)

            prev_score = scores_sorted[i]

        # Final point
        p = tp / max(tp + fp, 1)
        r = tp / n_pos
        precisions.append(p)
        recalls.append(r)

        # Trapezoidal integration
        auc = 0.0

        for i in range(
            1,
            len(recalls)
        ):

            auc += (
                (recalls[i] - recalls[i - 1])
                * (precisions[i] + precisions[i - 1])
                / 2.0
            )

        return float(auc)

    except Exception:

        return 0.0


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    dataset_name,
    model,
    selected_columns,
    label_column,
    normal_label,
    threshold,
    scores,
    predictions,
    y_test,
    y_true_binary,
    accuracy,
    precision,
    recall,
    f1,
    roc_auc,
    pr_auc,
    config,
    test_row_index=None,
    explanations=None,
    scorer=None,
    scorer_table=None
):
    """
    Save model, predictions and metadata.
    """

    output_path = os.path.join(
        SAFE_DIR,
        dataset_name
    )

    ensure_directory(
        output_path
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model_path = os.path.join(
        output_path,
        "safe_model.pt"
    )

    torch.save(
        model.state_dict(),
        model_path
    )

    if scorer is not None and scorer.lof is not None:

        with open(os.path.join(output_path, "safe_scorer.pkl"), "wb") as file:

            pickle.dump({"kind": scorer.kind, "n_neighbors": scorer.n_neighbors,
                         "scaler": scorer.scaler, "lof": scorer.lof}, file)

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    predictions_path = os.path.join(
        output_path,
        "safe_predictions.csv"
    )

    result_df = pd.DataFrame({

        "test_row_index":
            test_row_index if test_row_index is not None
            else np.full(len(scores), -1),

        "anomaly_score":
            scores,

        "true_original_label":
            y_test,

        "true_binary_label":
            y_true_binary,

        "predicted_binary_label":
            predictions,

        "verdict":
            np.where(
                predictions == 1,
                "Anomaly",
                "Normal"
            )
    })

    if explanations is not None:

        result_df["top_features"] = explanations

    result_df.to_csv(
        predictions_path,
        index=False
    )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    metadata = {

        "module":
            "M3 SAFE",

        "method":
            "Self-supervised Masked Autoencoder",

        "dataset":
            dataset_name,

        "device":
            str(DEVICE),

        "label_column":
            label_column,

        "normal_label":
            normal_label,

        "selected_feature_count":
            len(selected_columns),

        "selected_features":
            selected_columns,

        "mask_ratio":
            config["mask_ratio"],

        "epochs":
            config["epochs"],

        "batch_size":
            config["batch_size"],

        "learning_rate":
            config["lr"],

        "latent_dim":
            config["latent"],

        "dropout":
            config["dropout"],

        "loss":
            config["loss"],

        "hyperparameter_source":
            config["source"],

        "scorer":
            scorer.name if scorer is not None else "recon",

        "scorer_selection_validation_auc":
            scorer_table or [],

        "mask_ratio_selection_validation_auc":
            config.get("mask_selection", []),

        "flag_rate_per_class": {
            str(int(v)): float(predictions[y_test == v].mean())
            for v in np.unique(y_test)
        },

        "max_validation_fpr":
            MAX_VALIDATION_FPR,

        "threshold":
            threshold,

        "training_samples":
            MAX_NORMAL_TRAIN_SAMPLES,

        "test_samples":
            len(y_test),

        "accuracy":
            float(accuracy),

        "precision":
            float(precision),

        "recall":
            float(recall),

        "f1_score":
            float(f1),

        "roc_auc":
            float(roc_auc),

        "pr_auc":
            float(pr_auc),

        **imbalance_aware_metrics(
            int(np.sum((predictions == 1) & (y_true_binary == 1))),
            int(np.sum((predictions == 1) & (y_true_binary == 0))),
            int(np.sum((predictions == 0) & (y_true_binary == 0))),
            int(np.sum((predictions == 0) & (y_true_binary == 1)))
        ),

        "model_file":
            "safe_model.pt",

        "prediction_file":
            "safe_predictions.csv"
    }

    metadata_path = os.path.join(
        output_path,
        "safe_metadata.json"
    )

    with open(
        metadata_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metadata,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\nSAFE outputs saved:"
    )

    print(
        f"Model:       {model_path}"
    )

    print(
        f"Predictions: {predictions_path}"
    )

    print(
        f"Metadata:    {metadata_path}"
    )


# ============================================================
# PROCESS DATASET
# ============================================================

def process_dataset(
    dataset_name
):
    """
    Complete M3 SAFE pipeline.
    """

    print(
        "\n"
        + "=" * 65
    )

    print(
        f"M3 SAFE - {dataset_name}"
    )

    print(
        "=" * 65
    )

    # --------------------------------------------------------
    # Paths
    # --------------------------------------------------------

    dataset_dir = os.path.join(
        PROCESSED_DIR,
        dataset_name
    )

    train_path = os.path.join(
        dataset_dir,
        "train.csv"
    )

    test_path = os.path.join(
        dataset_dir,
        "test.csv"
    )

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        train_path
    ):

        raise FileNotFoundError(
            "\nTraining CSV not found:\n"
            f"{train_path}"
        )

    if not os.path.exists(
        test_path
    ):

        raise FileNotFoundError(
            "\nTest CSV not found:\n"
            f"{test_path}"
        )

    # --------------------------------------------------------
    # Detect label column
    # --------------------------------------------------------

    label_column = detect_label_column(
        train_path
    )

    print(
        f"\nDetected label column: "
        f"{label_column}"
    )

    # --------------------------------------------------------
    # Normal label
    # --------------------------------------------------------

    normal_label = determine_normal_label(
        dataset_name,
        train_path,
        label_column
    )

    print(
        f"Normal label: "
        f"{normal_label}"
    )

    # --------------------------------------------------------
    # Load WOA features
    # --------------------------------------------------------

    selected_features = load_selected_features(
        dataset_name
    )

    print(
        f"\nWOA selected features: "
        f"{len(selected_features)}"
    )

    # --------------------------------------------------------
    # Resolve feature names
    # --------------------------------------------------------

    selected_columns = resolve_feature_columns(
        train_path,
        selected_features
    )

    leaked = [c for c in selected_columns if c in LEAKY_COLUMNS]

    if leaked:

        print(
            f"\nWARNING: dropping label-leaking column(s) "
            f"from features: {leaked}"
        )

        selected_columns = [
            c for c in selected_columns
            if c not in LEAKY_COLUMNS
        ]

    print(
        f"Matched features: "
        f"{len(selected_columns)}"
    )

    # --------------------------------------------------------
    # Hyperparameters (tuned per dataset if available)
    # --------------------------------------------------------

    config = load_safe_config(
        dataset_name
    )

    print(
        f"\nHyperparameters ({config['source']}):"
    )

    for key in ["epochs", "mask_ratio", "lr", "batch_size", "latent", "dropout", "loss", "scorer"]:

        print(
            f"  {key}: {config[key]}"
        )

    # --------------------------------------------------------
    # Display selected features
    # --------------------------------------------------------

    print(
        "\nFirst selected features:"
    )

    display_count = min(
        10,
        len(selected_columns)
    )

    for feature in selected_columns[
        :display_count
    ]:

        print(
            f"  - {feature}"
        )

    if len(selected_columns) > 10:

        print(
            f"  ... and "
            f"{len(selected_columns) - 10} more"
        )

    # --------------------------------------------------------
    # Load ALL normal training samples
    # --------------------------------------------------------

    X_normal_all = load_normal_training_data(
        train_path,
        selected_columns,
        label_column,
        normal_label,
        MAX_NORMAL_TRAIN_SAMPLES
    )

    # --------------------------------------------------------
    # Split normal data: train + validation
    # --------------------------------------------------------

    n_total = len(X_normal_all)

    n_val = max(
        100,
        int(n_total * VALIDATION_FRACTION)
    )

    n_train = n_total - n_val

    X_normal_train = X_normal_all[:n_train]
    X_normal_val = X_normal_all[n_train:]

    print(
        f"\nNormal training split: "
        f"{n_train:,} train, "
        f"{n_val:,} validation"
    )

    # --------------------------------------------------------
    # Feature Standardization (fit ONLY on normal train)
    # --------------------------------------------------------
    print(
        "\nStandardizing feature distributions "
        "using RobustScaler (Median & IQR)..."
    )
    scaler = RobustScaler()
    X_normal_train = np.clip(scaler.fit_transform(X_normal_train), -10.0, 10.0)
    X_normal_val = np.clip(scaler.transform(X_normal_val), -10.0, 10.0)

    # --------------------------------------------------------
    # Load attack validation data (from TRAINING set)
    # --------------------------------------------------------

    X_attack_val = load_attack_validation_data(
        train_path,
        selected_columns,
        label_column,
        normal_label,
        MAX_ATTACK_VALIDATION_SAMPLES
    )

    if X_attack_val is not None and len(X_attack_val) > 0:
        X_attack_val = np.clip(scaler.transform(X_attack_val), -10.0, 10.0)

    # --------------------------------------------------------
    # Train SAFE
    # --------------------------------------------------------

    # One model per candidate mask ratio; each gets its scorer selected on
    # validation, and the (mask ratio, scorer) pair with the best VALIDATION
    # ROC-AUC is kept. A single mask ratio means no search.
    mask_candidates = (
        list(config["mask_ratio"])
        if isinstance(config["mask_ratio"], (list, tuple))
        else [config["mask_ratio"]]
    )

    best = None
    mask_selection = []

    for mask_ratio in mask_candidates:

        torch.manual_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        candidate_config = {**config, "mask_ratio": mask_ratio}

        if len(mask_candidates) > 1:

            print(f"\n----- Candidate mask ratio {mask_ratio} -----")

        model_i = train_safe_model(
            X_normal_train,
            X_normal_train.shape[1],
            candidate_config
        )

        # --------------------------------------------------------
        # Anomaly scorer (selected on validation data)
        # --------------------------------------------------------

        scorer_i, table_i = select_scorer(
            model_i,
            X_normal_train,
            X_normal_val,
            X_attack_val,
            config.get("scorer", SCORER)
        )

        auc_i = next(
            (r["val_roc_auc"] for r in table_i if r["scorer"] == scorer_i.name),
            float("nan")
        )

        mask_selection.append(
            {"mask_ratio": mask_ratio, "scorer": scorer_i.name, "val_roc_auc": auc_i}
        )

        if best is None or (
            not np.isnan(auc_i) and auc_i > best[3] + SCORER_TOLERANCE
        ):

            best = (mask_ratio, model_i, (scorer_i, table_i), auc_i)

    mask_ratio, model, (scorer, scorer_table), _ = best

    config = {**config, "mask_ratio": mask_ratio, "mask_selection": mask_selection}

    if len(mask_candidates) > 1:

        print("\nMask ratio selection (validation ROC-AUC of the selected scorer):")

        for r in mask_selection:

            print(f"  mask {r['mask_ratio']:<4} {r['scorer']:<18} {r['val_roc_auc']}")

        print(f"Selected mask ratio: {mask_ratio}  (scorer {scorer.name})")

    print(
        "\nCalculating validation "
        "anomaly scores..."
    )

    normal_val_errors = scorer.score(
        X_normal_val
    )

    # --------------------------------------------------------
    # Calculate threshold
    # --------------------------------------------------------

    if X_attack_val is not None and len(
        X_attack_val
    ) > 0:

        attack_val_errors = scorer.score(
            X_attack_val
        )

        # Print validation score statistics
        print_score_statistics(
            normal_val_errors,
            attack_val_errors
        )

        threshold = calculate_threshold_validation(
            normal_val_errors,
            attack_val_errors
        )

    else:

        # Fallback: percentile-based
        print(
            "\nNo attack validation data available."
        )

        print(
            "Using percentile-based threshold."
        )

        threshold = calculate_threshold_percentile(
            normal_val_errors
        )

    print(
        f"\nFinal SAFE threshold: "
        f"{threshold:.6f}"
    )

    # --------------------------------------------------------
    # Load test data (stratified)
    # --------------------------------------------------------

    X_test, y_test, test_row_index = load_test_data(
        test_path,
        selected_columns,
        label_column,
        normal_label,
        MAX_TEST_SAMPLES,
        return_index=True
    )

    # Transform test features using the fitted RobustScaler
    X_test = np.clip(scaler.transform(X_test), -10.0, 10.0)


    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    (
        scores,
        predictions,
        y_true_binary,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc
    ) = evaluate_safe(
        scorer,
        X_test,
        y_test,
        normal_label,
        threshold
    )

    # --------------------------------------------------------
    # Per-feature explanations (used as evidence by M4/M5)
    # --------------------------------------------------------

    explanations = explain_predictions(
        model,
        X_test,
        X_normal_val,
        selected_columns
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_results(
        dataset_name,
        model,
        selected_columns,
        label_column,
        normal_label,
        threshold,
        scores,
        predictions,
        y_test,
        y_true_binary,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        pr_auc,
        config,
        test_row_index=test_row_index,
        explanations=explanations,
        scorer=scorer,
        scorer_table=scorer_table
    )

    print(
        f"\nFinished SAFE: "
        f"{dataset_name}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 65
    )

    print(
        "M3 - SAFE"
    )

    print(
        "Self-supervised Masked Autoencoder"
    )

    print(
        "=" * 65
    )

    print(
        f"\nUsing device: "
        f"{DEVICE}"
    )

    ensure_directory(
        SAFE_DIR
    )

    datasets = [
        "IoMT-TrafficData",
        "MedSec-25"
    ]

    # --------------------------------------------------------
    # Process both datasets
    # --------------------------------------------------------

    for dataset_name in datasets:

        try:

            process_dataset(
                dataset_name
            )

        except Exception as error:

            print(
                "\n"
                + "=" * 65
            )

            print(
                f"ERROR processing "
                f"{dataset_name}"
            )

            print(
                "=" * 65
            )

            print(
                str(error)
            )

            print(
                "\nSAFE stopped because this dataset "
                "could not be processed."
            )

            raise

    # --------------------------------------------------------
    # Finished
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 65
    )

    print(
        "ALL DATASETS PROCESSED SUCCESSFULLY!"
    )

    print(
        "=" * 65
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()