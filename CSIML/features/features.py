"""Features module"""

import pandas as pd
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition


def make_features(data: pd.DataFrame, features: list, stats: list) -> pd.DataFrame:
    """Make features based on MatMiner

    Args:
        data (pd.DataFrame):2 columns, 'Material' and 'Experimental'.
        features (list): The features you want to use.
        stats (list): The statistical methods, such as 'maximum', 'minimum' and so on.

    Returns:
        pd.DataFrame: The data with desired features after filling None values.

    """
    # Obtain elements for the formula (Material columns)
    data = StrToComposition().featurize_dataframe(data, "Material")

    # Use magpie data from matminer to generate features
    builder = ElementProperty("magpie", features, stats)
    builder.feature_labels()
    magpie_df = builder.featurize_dataframe(data, col_id="composition")
    data = magpie_df.sort_values(by="Experimental").fillna(0)

    return data
