"""This is python script for preprocessing data"""
from typing import List, Tuple

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from scipy import stats
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PowerTransformer, RobustScaler


class DataPreprocess(BaseEstimator, TransformerMixin):
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

        if input_df.isnull().sum().any():
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

    def treat_skewness(self, fit_df: pd.DataFrame, transform_df: pd.DataFrame) -> pd.DataFrame:
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
        input_df: more gaussian like features of pandas dataframe
        """
        var = fit_df.columns
        skew_list = []
        for i in var:
            skew_list.append(fit_df[i].skew())

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
        power_transformer.fit(fit_df)
        gauss_df = power_transformer.transform(transform_df)
        return pd.DataFrame(gauss_df, columns=transform_df.columns)

    def check_skewed(self, input_df: pd.DataFrame) -> matplotlib.figure.Figure:
        """
        Checking if dataframe is skewed distributin

        Parameters
        ----------
        input_df: pd.DataFrame

        Return
        ------
        matplotlib.figure.Figure
        """
        var = input_df.columns

        with plt.style.context("dark_background"):
            _, axes = plt.subplots(10, 3, figsize=(10, 15), facecolor="m")
            axes = axes.flatten()

            for i, ax in enumerate(axes):
                if i < len(var):
                    sns.histplot(input_df[var[i]], ax=ax)
                    ax.set_title(var[i], fontsize=20)
                    ax.set_ylabel("Count", fontsize=20)  # set ylabel of the subplot
                    ax.tick_params(axis="both", labelsize=15)
                    ax.set_xlabel("")  # set empty string as x label of the subplot

            plt.tight_layout()
            plt.show()

    def check_imbalanced(self, input_df: pd.DataFrame) -> matplotlib.figure.Figure:
        """
        Printing out Imbalanced Class dataset

        Parameters
        ----------
        input_df: pandas dataframe

        Returns
        -------
        bar plot
        """
        temp = input_df["Class"].value_counts()
        df_class = pd.DataFrame({"Class": temp.index, "Value": temp.values})
        normal_num = df_class.loc[0, "Value"]
        fraud_num = df_class.loc[1, "Value"]
        print(
            f"Class 0: {normal_num/(normal_num+fraud_num)*100 : .2f}% \
            is normal transaction ({normal_num})"
        )
        print(
            f"Class 1: {fraud_num/(normal_num+fraud_num)*100 : .2f}% \
            is fraudulent transaction ({fraud_num})"
        )
        sns.barplot(x="Class", y="Value", data=df_class)
        plt.title("Class Distributions \n (0: No Fraud 1: Fraud)")

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
            df_sample = input_df.sample(frac=1)  # shuffling
            print(len(df_sample))
            fraud_df = df_sample.loc[df_sample["Class"] == 1]
            non_fraud_df = df_sample.loc[df_sample["Class"] == 0].sample(n=len(fraud_df))
            new_df = pd.concat([fraud_df, non_fraud_df]).sample(frac=1, random_state=42)

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

    def find_outlier(self, input_df: pd.DataFrame, mode: str):
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
            df_col = input_df[col].loc[input_df["Class"] == 1]
            med = np.median(df_col)
            ma = stats.median_abs_deviation(df_col)
            z = (0.6745 * (df_col - med)) / ma
            return list(z[np.abs(z) > 3].index)

        def _iqr(col):
            df_col = input_df[col].loc[input_df["Class"] == 1]
            q25, q75 = df_col.quantile(0.25), df_col.quantile(0.75)
            iqr = q75 - q25
            low_thresh = q25 - iqr * 1.5
            high_thresh = q75 + iqr * 1.5
            return input_df[(input_df[col] > high_thresh) | (input_df[col] < low_thresh)].index

        if mode.casefold() == "z-score":
            # this will be robust z-score aka median abs deviation method
            temp = list(map(lambda x: _mad(x), input_df))
            drop_idx = list(set(element for nested_list in temp for element in nested_list))
            return input_df.drop(index=drop_idx, inplace=True)

        elif mode.casefold() == "iqr":
            temp = list(map(lambda x: _iqr(x), input_df))
            drop_idx = list(set(element for nested_list in temp for element in nested_list))
            return input_df.drop(index=drop_idx, inplace=True)

        else:
            raise ValueError("mode should be either z-score, iqr, or dbscan")


if __name__ == "__main__":
    PATH = "../../data/raw/creditcard.csv"
    df = pd.read_csv(PATH)
    data_prep = DataPreprocess()

    (
        original_Xtrain,
        original_Xtest,
        original_ytrain,
        original_ytest,
    ) = data_prep.stratify_df(df, 0.1, ["Class"])
    # print(len(df))
    # print(len(original_Xtrain))
    # print(len(original_Xtest))

    # print(type(original_ytest))
    print(original_ytrain.value_counts())
    print(original_ytest.value_counts())

    # data_prep.check_imbalanced(df)

    # # df_under = data_prep.subsample(df, mode='undersampling')
    # df_over = data_prep.subsample(df, mode='oversampling')

    # X_clean = data_prep.cleaning(original_Xtrain)
    # X_scaled = data_prep.scaling(X_clean)
    # X_treat_skew = data_prep.treat_skewness(fit_df=X_scaled, transform_df=X_scaled)

    # # remove outlier, dont use it => rubbish
    # outlier_list=data_prep.find_outlier(df_over, mode='iqr')
    # print(len(outlier_list))
    # # for x,y in zip(df_under.columns,outlier_list):
    # #     print(f"{x} || {len(y)}")
