from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
import pandas as pd


def MakeFeatures(data: pd.DataFrame, features: list, stats: list):
    """
    :return: data, X, y
    """

    # Obtain elements for the formula (Material columns)
    data = StrToComposition().featurize_dataframe(data, "Material")

    # Use magpie data from matminer to generate features
    builder = ElementProperty("magpie", features, stats)
    builder.feature_labels()
    magpie_df = builder.featurize_dataframe(data, col_id="composition")
    data = magpie_df.sort_values(by="Experimental")
    X = data.iloc[:, 3:].fillna(0).values
    y = data.iloc[:, 1].values

    return data, X, y
