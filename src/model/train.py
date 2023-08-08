import argparse
import pickle
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from data.preprocess import DataPreprocess


def train_model(
    path: str,
    model: str,
    mode: str,
    model_path: str,
    grid_search: bool = False,
    grid_params: dict = None,
) -> pickle:
    """
    Training model

    Parameters
    ----------
    path : str
        input csv file path

    model: str
        ML model type e.g. rf, lr, xgboost, svm

    model_path: str
        output model pickle path

    grid_search: bool, default=False
        whether random grid serach is going to be used

    grid_params: dict, default=None
        grid search parameters specified by user

    Returns
    -------
    model pickle file dumped to given model path
    """
    data_prep = DataPreprocess()
    if path.split(".")[-1] == "csv":
        df = pd.read_csv(path)
        (original_Xtrain, _, original_ytrain, _) = data_prep.stratify_df(
            df, test_size=0.1, stratify=["Class"]
        )
        if mode == "base":
            # Baesline Model
            new_df = pd.concat([original_Xtrain, original_ytrain], axis=1)
            new_df = data_prep.scaling(data_prep.cleaning(new_df))
            X_train = new_df.drop("Class", axis=1)
            y_train = new_df["Class"]

        elif mode == "under":
            # Under-sampling Technique
            under_df = data_prep.subsample(
                pd.concat([original_Xtrain, original_ytrain], axis=1), mode="undersampling"
            )
            under_df = data_prep.scaling(data_prep.cleaning(under_df))
            X_train = under_df.drop("Class", axis=1)
            y_train = under_df["Class"]

        elif mode == "over":
            # Over-sampling Technique
            over_df = data_prep.subsample(
                pd.concat([original_Xtrain, original_ytrain], axis=1), mode="oversampling"
            )
            over_df = data_prep.scaling(data_prep.cleaning(over_df))
            X_train = over_df.drop("Class", axis=1)
            y_train = over_df["Class"]

    elif path.split(".")[-1] == "snappy":
        df = pd.read_parquet(path)
        if mode == "base":
            # Baesline Model
            X_train = df.drop("Class", axis=1)
            y_train = df["Class"]

        elif mode == "under":
            # Under-sampling Technique
            under_df = data_prep.subsample(df, mode="undersampling")
            X_train = under_df.drop("Class", axis=1)
            y_train = under_df["Class"]

        elif mode == "over":
            # Over-sampling Technique
            over_df = data_prep.subsample(df, mode="oversampling")
            X_train = over_df.drop("Class", axis=1)
            y_train = over_df["Class"]

    # Model Training
    start_time = time.time()
    if model == "rf":
        rf = RandomForestClassifier(random_state=42, class_weight=None, verbose=1)
        if grid_search:
            rf_params = grid_params
            rf_cv = GridSearchCV(estimator=rf, param_grid=rf_params, n_jobs=-1, verbose=1, cv=4)
            rf_cv.fit(X=X_train, y=y_train)
            best_model = rf_cv.best_estimator_
            print(
                pd.Series(data=best_model.feature_importances_, index=X_train.columns).sort_values(
                    ascending=False
                )
            )
            # Saving trained Model
            pickle.dump(best_model, open(model_path, "wb"))

        else:
            rf.fit(X_train, y_train)
            # Saving trained Model
            pickle.dump(rf, open(model_path, "wb"))
    print(f"time taken for training is {time.time() - start_time}")


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

    csv_path = "../data/raw/creditcard.csv"
    pq_path = "../data/feature/treat_skew.parquet.snappy"

    filename = f"rf_treat_skew_{arguments['mode']}.pkl"
    model_path = f"../data/trained_model/{filename}"
    rf_params = {
        "bootstrap": [True, False],
        "max_depth": [10, 20, 30],
        "min_samples_leaf": [1, 2, 4],
        "min_samples_split": [2, 5],
        "n_estimators": [200, 300],
    }

    train_model(
        path=pq_path,
        model="rf",
        mode=arguments["mode"],
        model_path=model_path,
        grid_search=False,
        grid_params=rf_params,
    )
