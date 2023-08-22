import numpy as np
# import pandas as pd
from dython.nominal import associations
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon
from sklearn.preprocessing import MinMaxScaler

from sdmetrics.single_column import StatisticSimilarity


def get_correlation_similarity_score(real_data, synthetic_data, metadata):
    """
    Calculate similarity scores between real and synthetic datasets.

    Parameters:
        real_data (pd.DataFrame): Real dataset.
        synthetic_data (pd.DataFrame): Synthetic dataset.
        metadata (dict): Additional metadata for computation.

    Returns:
        corr_dist (float): Similarity score based on correlation distance.

    Details:
        - Pearson's R for continuous-continuous cases
        - Correlation Ratio for categorical-continuous cases
        - Cramer's V or Theil's U for categorical-categorical cases
    """
    # TODO: Compute nominal and numerical columns from metadata
    nominal_columns = []  # List of nominal columns
    numerical_columns = []  # List of numerical columns

    # Compute correlation matrices for real and synthetic datasets
    real_corr = associations(real_data,
                             nominal_columns=nominal_columns,
                             nom_nom_assoc='theil',
                             nom_num_assoc='correlation_ratio',
                             num_num_assoc='pearson')

    fake_corr = associations(synthetic_data,
                             nominal_columns=nominal_columns,
                             nom_nom_assoc='theil',
                             nom_num_assoc='correlation_ratio',
                             num_num_assoc='pearson')

    # Calculate the Euclidean distance between the correlation matrices
    corr_dist = np.linalg.norm(real_corr - fake_corr)

    # TODO: Uncomment these lines if log-transformed correlations are needed
    # real_log_corr = np.sign(real_corr) * np.log(abs(real_corr))
    # fake_log_corr = np.sign(fake_corr) * np.log(abs(fake_corr))

    return corr_dist


def get_statistic_similarity_score(real_col, synthetic_col, statistic):
    """
    Calculate the statistic similarity score for a given column.

    Parameters:
        real_col (pandas.Series): Real data column.
        synthetic_col (pandas.Series): Synthetic data column.
        statistic (str): Statistic to compare ('mean', 'median', 'std').

    Returns:
        statistic_similarity_score (float): Similarity score for the specified statistic.
    """
    # Calculate the statistic similarity using StatisticSimilarity metric
    statistic_similarity_score = StatisticSimilarity.compute(
        real_data=real_col,
        synthetic_data=synthetic_col,
        statistic=statistic
    )

    return statistic_similarity_score


def get_distance_score(real_col, synthetic_col, col_data_type):
    """
    Compute the appropriate distance (Wasserstein or Jensen-Shannon) based on data type.

    Parameters:
        real_col (pandas.Series): The real dataset.
        synthetic_col (pandas.Series): The synthetic dataset.
        col_data_type (str): Data type of the column.

    Returns:
        distance (float): The calculated distance.
    """
    # Drop missing values from both datasets
    real_cleaned = real_col.dropna()
    synthetic_cleaned = synthetic_col.dropna()

    if col_data_type in ["categorical", "numerical"]:
        # Categorical data: Use Jensen-Shannon distance
        categories = real_cleaned.unique()
        real_counts = real_cleaned.value_counts()
        synthetic_counts = synthetic_cleaned.value_counts()

        # Calculate probability density functions (PDFs)
        real_pdf = real_counts / len(real_cleaned)
        synthetic_pdf = synthetic_counts / len(synthetic_cleaned)

        categories = real_pdf.index.tolist()
        real_pdf_values = real_pdf.values.tolist()
        synthetic_pdf_values = [
            synthetic_pdf.get(cat, 0) for cat in categories]

        # Compute Jensen-Shannon distance
        js_distance = jensenshannon(real_pdf_values, synthetic_pdf_values, 2.0)
        return js_distance
    else:
        # Numerical data: Use Wasserstein distance
        scaler = MinMaxScaler()

        # Normalize both datasets using the same scaler
        real_scaled = scaler.fit_transform(
            real_cleaned.values.reshape(-1, 1)).flatten()
        synthetic_scaled = scaler.transform(
            synthetic_cleaned.values.reshape(-1, 1)).flatten()

        # Compute Wasserstein distance for normalized datasets
        w_distance = wasserstein_distance(real_scaled, synthetic_scaled)
        return w_distance
