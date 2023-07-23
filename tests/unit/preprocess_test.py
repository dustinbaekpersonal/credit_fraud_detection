import pandas as pd
from pandas.testing import assert_series_equal

from src.data.preprocess import DataPreprocess

input_df = pd.DataFrame({"V1": list(range(1, 21)), "Class": [0] * 10 + [1] * 10})

expected_y_train = pd.Series([9, 9], index=[0, 1], name="count")
expected_y_train.index.name = "Class"

expected_y_test = pd.Series([1, 1], index=[0, 1], name="count")
expected_y_test.index.name = "Class"


def test_stratify_df():
    """Unit tests for stratify_df"""
    data_preprocess = DataPreprocess()

    _, _, y_train, y_test = data_preprocess.stratify_df(
        input_df=input_df, test_size=0.1, stratify="Class"
    )
    assert_series_equal(y_train.value_counts().sort_index(), expected_y_train.sort_index())
    assert_series_equal(y_test.value_counts().sort_index(), expected_y_test.sort_index())
