import numpy as np
# import pandas as pd
from dython.nominal import associations
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
from sdmetrics.single_column import StatisticSimilarity
from sklearn.preprocessing import MinMaxScaler


def compute_correlation_similarity(real_data, synthetic_data, cols):
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
        - Ref: https://shakedzy.xyz/dython/modules/nominal/
    """
    # TODO: Compute nominal and numerical columns from metadata
    nominal_columns = cols["categorical"]  # List of nominal columns
    # numerical_columns = []  # List of numerical columns

    # Compute correlation matrices for real and synthetic datasets
    real_corr = associations(real_data,
                             nominal_columns=nominal_columns,
                             nom_nom_assoc='theil',
                             nom_num_assoc='correlation_ratio',
                             num_num_assoc='pearson',
                             plot=False)

    # {'corr': , 'ax': <Axes: >}
    synthetic_corr = associations(synthetic_data,
                                  nominal_columns=nominal_columns,
                                  nom_nom_assoc='theil',
                                  nom_num_assoc='correlation_ratio',
                                  num_num_assoc='pearson',
                                  plot=False)

    # Calculate the Euclidean distance between the correlation matrices
    corr_dist = np.linalg.norm(real_corr['corr'] - synthetic_corr['corr'])

    # TODO: Uncomment these lines if log-transformed correlations are needed
    # real_log_corr = np.sign(real_corr) * np.log(abs(real_corr))
    # synthetic_log_corr = np.sign(synthetic_corr) * np.log(abs(synthetic_corr))

    return corr_dist, real_corr["ax"], synthetic_corr["ax"]


def compute_statistic_similarity(real_col, synthetic_col, statistic):
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


# import pandas as pd
# from scipy.spatial.distance import jensenshannon
# import numpy as np

def compute_jensenshannon_distance(real_col, synthetic_col, col_data_type):
    """
    Compute Jensen-Shannon distance for both categorical and numerical data.

    Parameters:
        real_col (pandas.Series): The real dataset.
        synthetic_col (pandas.Series): The synthetic dataset.
        col_data_type (str): Data type of the column ('categorical' or 'numerical').

    Returns:
        distance (float): The calculated Jensen-Shannon distance.
    """
    # Drop missing values from both datasets
    real_cleaned = real_col.dropna()
    synthetic_cleaned = synthetic_col.dropna()

    if col_data_type == "categorical":
        # Calculate probability distributions for categorical data
        real_pdf = real_cleaned.value_counts(normalize=True)
        synthetic_pdf = synthetic_cleaned.value_counts(normalize=True)

        # Align categories
        all_categories = set(real_pdf.index).union(synthetic_pdf.index)
        real_pdf = real_pdf.reindex(all_categories, fill_value=0)
        synthetic_pdf = synthetic_pdf.reindex(all_categories, fill_value=0)

    elif col_data_type == "numerical":
        # Define bins for numerical data
        bins = np.histogram_bin_edges(np.concatenate((real_cleaned, synthetic_cleaned)), bins='auto')
        
        # Calculate histograms and normalize to create probability distributions
        real_hist, _ = np.histogram(real_cleaned, bins=bins, density=True)
        synthetic_hist, _ = np.histogram(synthetic_cleaned, bins=bins, density=True)

        # Handle cases where histograms have all zeros
        real_hist[real_hist == 0] = 1e-8
        synthetic_hist[synthetic_hist == 0] = 1e-8

        real_pdf = real_hist / real_hist.sum()
        synthetic_pdf = synthetic_hist / synthetic_hist.sum()

    else:
        raise ValueError("Invalid data type. Please specify 'categorical' or 'numerical'.")

    # Compute Jensen-Shannon distance
    js_distance = jensenshannon(real_pdf, synthetic_pdf)
    return js_distance

# # Example Usage
# real_data_categorical = pd.Series(['A', 'B', 'A', 'C'])
# synthetic_data_categorical = pd.Series(['A', 'B', 'B', 'C'])

# real_data_numerical = pd.Series([1, 2, 3, 4, 5])
# synthetic_data_numerical = pd.Series([2, 2, 3, 4, 5])

# js_cat = compute_jensenshannon_distance(real_data_categorical, synthetic_data_categorical, "categorical")
# js_num = compute_jensenshannon_distance(real_data_numerical, synthetic_data_numerical, "numerical")

# print(f"Jensen-Shannon Distance (Categorical): {js_cat}")
# print(f"Jensen-Shannon Distance (Numerical): {js_num}")


# def compute_jensenshannon_distance(real_col, synthetic_col, col_data_type):
#     """
#     Compute the appropriate Jensen-Shannon distance

#     Parameters:
#         real_col (pandas.Series): The real dataset.
#         synthetic_col (pandas.Series): The synthetic dataset.
#         col_data_type (str): Data type of the column.

#     Returns:
#         distance (float): The calculated distance.
#     """
#     # Drop missing values from both datasets
#     real_cleaned = real_col.dropna()
#     synthetic_cleaned = synthetic_col.dropna()

#     # if col_data_type in ["categorical"]: #, "numerical"]:

#     # Categorical data: Use Jensen-Shannon distance
#     categories = real_cleaned.unique()
#     real_counts = real_cleaned.value_counts()
#     synthetic_counts = synthetic_cleaned.value_counts()

#     # Calculate probability density functions (PDFs)
#     real_pdf = real_counts / len(real_cleaned)
#     synthetic_pdf = synthetic_counts / len(synthetic_cleaned)

#     categories = real_pdf.index.tolist()
#     real_pdf_values = real_pdf.values.tolist()
#     synthetic_pdf_values = [
#         synthetic_pdf.get(cat, 0) for cat in categories]

#     # Compute Jensen-Shannon distance
#     js_distance = jensenshannon(real_pdf_values, synthetic_pdf_values, 2.0)
#     return js_distance


def compute_wasserstein_distance(real_col, synthetic_col, col_data_type):
    """
    Compute the Wasserstein distance based on data type.

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

    scaler = MinMaxScaler()

    # Normalize both datasets using the same scaler
    real_scaled = scaler.fit_transform(
        real_cleaned.values.reshape(-1, 1)).flatten()
    synthetic_scaled = scaler.transform(
        synthetic_cleaned.values.reshape(-1, 1)).flatten()

    # Compute Wasserstein distance for normalized datasets
    w_distance = wasserstein_distance(real_scaled, synthetic_scaled)
    return w_distance


# def compute_distance(real_col, synthetic_col, col_data_type):
#     """
#     Compute the appropriate distance (Wasserstein or Jensen-Shannon) based on data type.

#     Parameters:
#         real_col (pandas.Series): The real dataset.
#         synthetic_col (pandas.Series): The synthetic dataset.
#         col_data_type (str): Data type of the column.

#     Returns:
#         distance (float): The calculated distance.
#     """
#     # Drop missing values from both datasets
#     real_cleaned = real_col.dropna()
#     synthetic_cleaned = synthetic_col.dropna()

#     if col_data_type in ["categorical"]: #, "numerical"]:
#         # Categorical data: Use Jensen-Shannon distance
#         categories = real_cleaned.unique()
#         real_counts = real_cleaned.value_counts()
#         synthetic_counts = synthetic_cleaned.value_counts()

#         # Calculate probability density functions (PDFs)
#         real_pdf = real_counts / len(real_cleaned)
#         synthetic_pdf = synthetic_counts / len(synthetic_cleaned)

#         categories = real_pdf.index.tolist()
#         real_pdf_values = real_pdf.values.tolist()
#         synthetic_pdf_values = [
#             synthetic_pdf.get(cat, 0) for cat in categories]

#         # Compute Jensen-Shannon distance
#         js_distance = jensenshannon(real_pdf_values, synthetic_pdf_values, 2.0)
#         return js_distance
#     else:
#         # Numerical data: Use Wasserstein distance
#         scaler = MinMaxScaler()

#         # Normalize both datasets using the same scaler
#         real_scaled = scaler.fit_transform(
#             real_cleaned.values.reshape(-1, 1)).flatten()
#         synthetic_scaled = scaler.transform(
#             synthetic_cleaned.values.reshape(-1, 1)).flatten()

#         # Compute Wasserstein distance for normalized datasets
#         w_distance = wasserstein_distance(real_scaled, synthetic_scaled)
#         return w_distance
