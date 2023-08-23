"""Privacy metrics"""
import numpy as np
from sdmetrics.single_table import NewRowSynthesis
from sklearn import metrics
from sklearn.preprocessing import StandardScaler


def compute_new_row_synthesis(real_data, synthetic_data, metadata, 
                                    numerical_match_tolerance=0.01, 
                                    synthetic_sample_size=None):
    """
    Compute the New Row Synthesis score between real and synthetic datasets.

    Parameters:
        real_data (pd.DataFrame): Real dataset.
        synthetic_data (pd.DataFrame): Synthetic dataset.
        metadata (dict): Metadata describing the columns.
        numerical_match_tolerance (float, optional): Tolerance for numerical value matching.
            Defaults to 0.01 (1%).
        synthetic_sample_size (int, optional): Number of synthetic rows to sample before computing.
            If None, use all synthetic data. Defaults to None.

    Returns:
        new_row_synthesis_score (float): The computed New Row Synthesis score.
    """
    # Calculate the New Row Synthesis score using the metric
    new_row_synthesis_score = NewRowSynthesis.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        numerical_match_tolerance=numerical_match_tolerance,
        synthetic_sample_size=synthetic_sample_size
    )
    
    return new_row_synthesis_score


def _compute_nearest_neighbor_metrics(distances):
    """
    Calculate nearest neighbor ratios and fifth percentiles of distances.

    Parameters:
        distances (numpy.ndarray): Pairwise distances between data points.

    Returns:
        nn_ratios (numpy.ndarray): Nearest neighbor ratios.
        fifth_percentiles (numpy.ndarray): Fifth percentiles of distances.
    """
    smallest_two_indexes = [dist.argsort()[:2] for dist in distances]
    smallest_two_distances = [dist[indices] for dist,
                              indices in zip(distances, smallest_two_indexes)]
    nn_ratios = np.array([dist[0] / dist[1]
                         for dist in smallest_two_distances])
    fifth_percentiles = np.percentile(smallest_two_distances, 5, axis=1)
    return nn_ratios, fifth_percentiles


def compute_distance_to_closest_records(real_data, synthetic_data, data_percent=15):
    """
    Calculate distance to closest records metrics.

    Parameters:
        real_path (str): Path to the real dataset CSV file.
        fake_path (str): Path to the synthetic dataset CSV file.
        data_percent (int): Percentage of data to sample.

    Returns:
        results (dict): Dictionary containing calculated metrics.
    """
    data_scaling_factor = 0.01

    # Load and preprocess real and fake datasets
    real = real_data.drop_duplicates(keep=False)
    fake = synthetic_data.drop_duplicates(keep=False)

    num_samples = int(len(real) * (data_scaling_factor * data_percent))
    real_refined = real.sample(n=num_samples, random_state=42).to_numpy()
    fake_refined = fake.sample(n=num_samples, random_state=42).to_numpy()

    scalerR = StandardScaler()
    scalerF = StandardScaler()
    df_real_scaled = scalerR.fit_transform(real_refined)
    df_fake_scaled = scalerF.fit_transform(fake_refined)

    # Calculate pairwise distances
    dist_rf = metrics.pairwise_distances(
        df_real_scaled, Y=df_fake_scaled, metric='minkowski', n_jobs=-1)
    dist_rr = metrics.pairwise_distances(
        df_real_scaled, Y=None, metric='minkowski', n_jobs=-1)
    dist_ff = metrics.pairwise_distances(
        df_fake_scaled, Y=None, metric='minkowski', n_jobs=-1)

    # Calculate nearest neighbor metrics
    nn_ratios_rf, fifth_perc_rf = _compute_nearest_neighbor_metrics(dist_rf)
    nn_ratios_rr, fifth_perc_rr = _compute_nearest_neighbor_metrics(dist_rr)
    nn_ratios_ff, fifth_perc_ff = _compute_nearest_neighbor_metrics(dist_ff)

    # Store results in a dictionary
    results = {
        "nn_ratio_rf": nn_ratios_rf,
        "fifth_perc_rf": fifth_perc_rf,
        "nn_ratio_rr": nn_ratios_rr,
        "fifth_perc_rr": fifth_perc_rr,
        "nn_ratio_ff": nn_ratios_ff,
        "fifth_perc_ff": fifth_perc_ff
    }

    return results

# import pandas as pd
# import numpy as np
# from sklearn.metrics.pairwise import euclidean_distances

# def distance_to_closest_records(synthetic_data, real_data):
#     distances = euclidean_distances(synthetic_data, real_data)
#     min_distances = np.min(distances, axis=1)

#     # Normalize the distances to a range of 0 to 1
#     normalized_distances = (min_distances - np.min(min_distances)) / (np.max(min_distances) - np.min(min_distances))

#     return normalized_distances


# # Convert categorical features to numeric using one-hot encoding
# synthetic_data = pd.get_dummies(synthetic_data)
# real_data = pd.get_dummies(real_data)

# synthetic_array = synthetic_data.to_numpy()
# real_array = real_data.to_numpy()

# normalized_distances = distance_to_closest_records(synthetic_array, real_array)
# print("Normalized distances to closest real records:", normalized_distances)


# https://docs.sdv.dev/sdmetrics/metrics/metrics-glossary/categoricalcap
# from sdmetrics.single_table import CategoricalCAP
# score = CategoricalCAP.compute(
#     real_data=real_table,
#     synthetic_data=synthetic_table,
#     key_fields=['age_bracket', 'gender'],
#     sensitive_fields=['political_affiliation']
# )
