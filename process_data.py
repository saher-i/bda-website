import os

import numpy as np
import pandas as pd


def process_ieee_cis_data():
    # Create output directory
    if not os.path.exists("data/ieee_cis_clean"):
        print("Creating output directory...")
        os.makedirs("data/ieee_cis_clean")

    # Load data
    print("Loading data...")
    transaction_df = pd.read_csv("data/ieee_cis/train_transaction.csv")
    identity_df = pd.read_csv("data/ieee_cis/train_identity.csv")
    test_transaction = pd.read_csv("data/ieee_cis/test_transaction.csv")
    test_identity = pd.read_csv("data/ieee_cis/test_identity.csv")

    # Define column groups
    id_cols = [
        "card1",
        "card2",
        "card3",
        "card4",
        "card5",
        "card6",
        "ProductCD",
        "addr1",
        "addr2",
        "P_emaildomain",
        "R_emaildomain",
    ]
    cat_cols = ["M1", "M2", "M3", "M4", "M5", "M6", "M7", "M8", "M9"]

    # Split into train/test
    train_data_ratio = 0.8
    n_train = int(transaction_df.shape[0] * train_data_ratio)
    test_ids = transaction_df.TransactionID.values[n_train:]

    # Print fraud statistics
    get_fraud_frac = lambda series: 100 * series.sum() / len(series)
    print(
        f"Percent fraud for train transactions: {get_fraud_frac(transaction_df.isFraud[:n_train]):.2f}"
    )
    print(
        f"Percent fraud for test transactions: {get_fraud_frac(transaction_df.isFraud[n_train:]):.2f}"
    )
    print(
        f"Percent fraud for all transactions: {get_fraud_frac(transaction_df.isFraud):.2f}"
    )

    # Save test IDs
    with open("data/ieee_cis_clean/test.csv", "w") as f:
        f.writelines(map(lambda x: str(x) + "\n", test_ids))

    # Define feature columns
    non_feature_cols = ["isFraud", "TransactionDT"] + id_cols
    feature_cols = [
        col for col in transaction_df.columns if col not in non_feature_cols
    ]

    # Process features
    features = pd.get_dummies(transaction_df[feature_cols], columns=cat_cols).fillna(0)
    features["TransactionAmt"] = features["TransactionAmt"].apply(np.log10)

    # Save processed data
    features.to_csv("data/ieee_cis_clean/features.csv", index=False, header=False)
    transaction_df[["TransactionID", "isFraud"]].to_csv(
        "data/ieee_cis_clean/tags.csv", index=False
    )

    # Process edge information
    edge_types = id_cols + list(identity_df.columns)
    all_id_cols = ["TransactionID"] + id_cols

    # Merge transaction and identity data
    full_identity_df = transaction_df[all_id_cols].merge(
        identity_df, on="TransactionID", how="left"
    )

    # Create edge lists for each feature
    edges = {}
    for etype in edge_types:
        edgelist = full_identity_df[["TransactionID", etype]].dropna()
        output_path = f"data/ieee_cis_clean/relation_{etype}_edgelist.csv"
        edgelist.to_csv(output_path, index=False, header=True)
        edges[etype] = edgelist

    return edges


if __name__ == "__main__":
    edges = process_ieee_cis_data()
    print("Data processing completed successfully")
