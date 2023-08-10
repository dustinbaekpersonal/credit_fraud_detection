import argparse
import pickle

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

from data.preprocess import DataPreprocess


def predict_model(
    path: str,
    model_path: str,
):
    best_model = pickle.load(open(model_path, "rb"))

    feature_impt = best_model.feature_importances_

    df_raw = pd.read_csv(path)

    data_prep = DataPreprocess()
    (_, original_Xtest, _, original_ytest) = data_prep.stratify_df(
        df_raw, test_size=0.2, stratify=["Class"]
    )

    test_df = data_prep.scaling(
        data_prep.cleaning(pd.concat([original_Xtest, original_ytest], axis=1))
    )
    X_test = test_df.drop("Class", axis=1)
    y_test = test_df["Class"]

    print(pd.Series(data=feature_impt, index=X_test.columns).sort_values(ascending=False))

    y_pred = best_model.predict(X_test)
    y_true = y_test

    target_names = ["class 0", "class 1"]
    print(
        f"Classification report is: \n \
        {classification_report(y_true=y_true, y_pred=y_pred, target_names=target_names)}"
    )
    print(f"Confusion matrix is: \n {confusion_matrix(y_true=y_true, y_pred=y_pred)}")
    print(f"Average Precision Score: {average_precision_score(y_true=y_true, y_score=y_pred):.3f}")
    print(f"Recall score is {recall_score(y_true=y_true, y_pred=y_pred):.3f}")
    df_eval = pd.DataFrame({"pred": y_pred, "true": y_true})
    print(
        "List of wrong predictions: \n"
        + df_eval[
            (df_eval["true"] == 1) & (df_eval["pred"] == 0)
            | (df_eval["true"] == 0) & (df_eval["pred"] == 1)
        ].to_string()
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        help="mode of subsampling: base, under, over",
        choices=["base", "under", "over"],
        type=str,
        default="base",
    )
    args = parser.parse_args()
    arguments = args.__dict__

    # PATH = "../data/raw/creditcard.csv"
    PATH = "../data/feature/clean_scaled.parquet.snappy"

    FILENAME = f"rf_clean_scaled_{arguments['mode']}.pkl"
    MODEL_PATH = f"../data/trained_model/{FILENAME}"
    predict_model(PATH, MODEL_PATH)
