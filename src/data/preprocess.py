"""This is python script for preprocessing data"""
from typing import List, Tuple

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler


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

    def treat_skewness(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply a power transform featurewise to make data more Gaussian-like.
        Power transforms are a family of parametric,
        monotonic transformations that are applied to make data more Gaussian-like.
        This is useful for modeling issues related to heteroscedasticity (non-constant variance),
        or other situations where normality is desired.

        Parameters
        ----------
        input_df: pandas dataframe

        Returns
        -------
        df: more gaussian like features of pandas dataframe
        """
        var = input_df.columns
        skew_list = []
        for i in var:
            skew_list.append(input_df[i].skew())

        tmp = pd.concat(
            [
                pd.DataFrame(var, columns=["Features"]),
                pd.DataFrame(skew_list, columns=["Skewness"]),
            ],
            axis=1,
        )
        tmp.set_index("Features", inplace=True)
        skewed_features = tmp.loc[(tmp["Skewness"] > 1) | (tmp["Skewness"] < -1)].index.tolist()
        print(f"Following features are skewed: {skewed_features}")
        power_transformer = PowerTransformer()
        gauss_df = power_transformer.fit_transform(input_df)
        return pd.DataFrame(gauss_df, columns=input_df.columns, index=input_df.index)

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

    def drop_outlier(self, input_df: pd.DataFrame, mode: str):
        """
        Don't Use it at the moment, not working

        Parameters
        ----------
        input_df: input dataframe
        mode: robust z-score method, IQR method, DBSCAN

        Returns
        -------
        input_df: pandas dataframe without outliers
        """

        def _mad(col):
            df_col = input_df[col].loc[input_df["Class"] == 0]
            med = np.median(df_col)
            ma = stats.median_abs_deviation(df_col)
            z = (0.6745 * (df_col - med)) / ma
            return list(z[np.abs(z) > 3].index)

        def _iqr(col):
            df = input_df[input_df["Class"] == 0]
            df_col = df[col]
            q25, q75 = df_col.quantile(0.25), df_col.quantile(0.75)
            iqr = q75 - q25
            low_thresh = q25 - iqr * 3
            high_thresh = q75 + iqr * 3
            df = df[(df[col] > high_thresh) | (df[col] < low_thresh)]
            return df.index

        if mode.casefold() == "z-score":
            # this will be robust z-score aka median abs deviation method
            temp = list(map(lambda x: _mad(x), input_df))
            drop_idx = list(set(element for nested_list in temp for element in nested_list))
            return input_df.drop(index=drop_idx, inplace=True)

        elif mode.casefold() == "iqr":
            temp = list(map(lambda x: _iqr(x), input_df))[:-1]
            drop_idx = list(set(element for nested_list in temp for element in nested_list))
            df = input_df.drop(index=drop_idx)
            return df, drop_idx

        else:
            raise ValueError("mode should be either z-score, iqr, or dbscan")

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
    df, drop_idx = data_prep.drop_outlier(df, mode="iqr")
    OUTPUT_PATH = "../data/feature/outlier_removed.parquet.snappy"
    data_prep.save_parquet(df, OUTPUT_PATH)
