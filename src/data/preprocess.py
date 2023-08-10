"""This is python script for preprocessing data"""
from typing import List, Tuple

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler


class DataPreprocess:
    """
    Contains different Data Preprocessing methods
    """

    def cleaning(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        If there is null value, imputation is implemented

        Parameters
        ----------
        input_df: pandas dataframe
            raw dataframe

        Returns
        -------
        df_imputed: imputed pandas dataframe
        """

        if input_df.isnull().sum().any() or input_df.isna().sum().any():
            fill_nan = SimpleImputer(missing_values="NaN", strategy="mean")
            df_imputed = pd.DataFrame(fill_nan.fit_transform(input_df))
            return df_imputed
        print("No NaN values, no need for imputation")
        return input_df

    def scaling(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply robust scaling to un-normalized columns

        Parameters
        ----------
        input_df: pandas dataframe

        Returns
        -------
        input_df: scaled pandas dataframe
        """
        # Normalising un-normalized features
        rob_scaler = RobustScaler()
        input_df["scaled_time"] = rob_scaler.fit_transform(input_df["Time"].values.reshape(-1, 1))
        input_df["scaled_amount"] = rob_scaler.fit_transform(
            input_df["Amount"].values.reshape(-1, 1)
        )
        input_df.drop(["Amount", "Time"], axis=1, inplace=True)
        return input_df

    def stratify_df(
        self, input_df: pd.DataFrame, test_size: float, stratify: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Creates stratified train test splits of data

        Parameters
        ----------
        input_df: pandas dataframe
        test_size: proportion of data to include in test set
        stratify: list of columns to stratify on

        Returns
        -------
        X_train, X_test, y_train, y_test: stratified split of data
        """
        X_train, X_test, y_train, y_test = train_test_split(
            input_df.drop(columns=stratify),
            input_df[stratify],
            test_size=test_size,
            stratify=input_df[stratify],
            random_state=42,
        )
        return X_train, X_test, y_train, y_test

    def subsample(self, input_df: pd.DataFrame, mode: str = "undersampling") -> pd.DataFrame:
        """
        Parameters
        ----------
        input_df: pandas dataframe
        mode: either "undersampling" or "oversampling"

        Returns
        -------
        new_df: subsampeld dataframe
        """

        if mode.casefold() == "undersampling":
            # Random under sampling of majority class
            fraud_df = input_df[input_df["Class"] == 1]
            non_fraud_df = input_df[input_df["Class"] == 0].sample(n=len(fraud_df))
            new_df = pd.concat([fraud_df, non_fraud_df])
            return new_df

        elif mode.casefold() == "oversampling":
            # SMOTE to oversample minority class
            smt = SMOTE()
            X_train_over, y_train_over = smt.fit_resample(
                input_df.drop("Class", axis=1), input_df["Class"]
            )
            new_df = pd.concat([X_train_over, y_train_over], axis=1)
            return new_df

        else:
            raise ValueError("mode should be undersampling or oversampling")

    def preprocess(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Combining all the preprocess steps. So that we only need to use this attribute.
        """
        df = self.cleaning(input_df)
        df = self.scaling(df)
        return df

    def save_parquet(self, input_df: pd.DataFrame, parquet_path: str):
        """
        saving preprocessed data to data/feature folder

        Parameters
        ----------
        input_df: pandas dataframe
        parquet_path: str, file path to data/feature folder

        Returns
        ------
        saved parquet file to data/feature folder
        """
        input_df.to_parquet(path=parquet_path, engine="auto", compression="snappy")


if __name__ == "__main__":
    PATH = "../data/raw/creditcard.csv"
    df = pd.read_csv(PATH)
    data_prep = DataPreprocess()

    (
        original_Xtrain,
        _,
        original_ytrain,
        _,
    ) = data_prep.stratify_df(df, 0.1, ["Class"])

    df = data_prep.preprocess(original_Xtrain)
    df = pd.concat([df, original_ytrain], axis=1)
    OUTPUT_PATH = "../data/feature/.parquet.snappy"
    data_prep.save_parquet(df, OUTPUT_PATH)
