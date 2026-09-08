import pandas as pd


DATASETS = {
    "IoMT-TrafficData": {
        "train": "data/processed/IoMT-TrafficData/train.csv",
        "test": "data/processed/IoMT-TrafficData/test.csv",
        "label": "is_attack"
    },

    "MedSec-25": {
        "train": "data/processed/MedSec-25/train.csv",
        "test": "data/processed/MedSec-25/test.csv",
        "label": "Label"
    }
}


for name, settings in DATASETS.items():

    print("\n" + "=" * 60)
    print(name)
    print("=" * 60)

    train = pd.read_csv(settings["train"])
    test = pd.read_csv(settings["test"])

    label = settings["label"]

    print("\nTrain shape:")
    print(train.shape)

    print("\nTest shape:")
    print(test.shape)

    print("\nTrain labels:")
    print(train[label].value_counts())

    print("\nTest labels:")
    print(test[label].value_counts())

    # Check feature values
    features = train.drop(columns=[label])

    print("\nMinimum feature value:")
    print(features.min().min())

    print("\nMaximum feature value:")
    print(features.max().max())

    print("\nMissing values:")
    print(train.isnull().sum().sum())

    print("\nVerification complete.")