import numpy as np
import pandas as pd


def add_time_features(df: pd.DataFrame, time_col: str = 'time') -> pd.DataFrame:
    """
    Add time-based features to dataframe.

    Args:
        df: DataFrame with a datetime column.
        time_col: Name of the datetime column.

    Returns:
        DataFrame with added time features.
    """
    df = df.copy()
    dt = pd.to_datetime(df[time_col])

    # Month (1-12)
    df['month'] = dt.dt.month

    # Day of year (1-365)
    day_of_year = dt.dt.dayofyear

    # Cyclical encoding for seasonal patterns
    df['season_sin'] = np.sin(2 * np.pi * day_of_year / 365)
    df['season_cos'] = np.cos(2 * np.pi * day_of_year / 365)

    return df


TIME_FEATURES = ['month', 'season_sin', 'season_cos']
