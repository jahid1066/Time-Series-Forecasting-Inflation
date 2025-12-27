import pandas as pd
import numpy as np

def load_and_prepare_data(filepath):
    """
    Load CPI data and compute monthly inflation using log differences.
    """
    df = pd.read_csv(filepath)
    df['observation_date'] = pd.to_datetime(df['observation_date'])
    df.set_index('observation_date', inplace=True)

    # Compute monthly inflation
    df['inflation'] = np.log(df['CPIAUCSL']).diff() * 100
    df.dropna(inplace=True)

    return df
