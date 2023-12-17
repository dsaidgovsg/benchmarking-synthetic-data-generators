"""Privacy metrics"""
import pandas as pd
import numpy as np
from sdmetrics.single_table import NewRowSynthesis
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

# from sklearn.metrics import pairwise_distances
# from sklearn.preprocessing import StandardScaler


def compute_new_row_synthesis(real_data, synthetic_data, metadata,
                              numerical_match_tolerance=0.01,
                              synthetic_sample_percent=None):
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
    #   numerical_match_tolerance: A float >0.0 representing how close two numerical values have to be in order to be considered a match.
    #   (default) 0.01, which represents 1%
    #   synthetic_sample_size: The number of synthetic rows to sample before computing this metric. Use this to speed up the computation time if you have a large amount of synthetic data. Note that the final score may not be as precise if your sample size is low.

    # Calculate the New Row Synthesis score using the metric
    
    print("~"*20)
    print(int(len(synthetic_data)*synthetic_sample_percent))
    print("~"*20)

    print(real_data)
    print(synthetic_data)

    new_row_synthesis_score = NewRowSynthesis.compute(
        real_data=real_data,
        synthetic_data=synthetic_data,
        metadata=metadata,
        numerical_match_tolerance=numerical_match_tolerance,
        synthetic_sample_size=int(len(synthetic_data)*synthetic_sample_percent)
    )

    print(f"new_row_synthesis_score: {new_row_synthesis_score}")
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


def compute_dcr_metrics(train_data, test_data, synthetic_data):
    # TODO: revisit
    """
    Calculate Distance to Closest Record (DCR) metrics.

    Parameters:
        train_data (numpy.ndarray): Training dataset.
        test_data (numpy.ndarray): Test dataset.
        synthetic_data (numpy.ndarray): Synthetic dataset.

    Returns:
        results (dict): Dictionary containing DCR metrics and privacy score.
    """
    # Calculate DCR for Test-Train
    dcr_test_train = np.min(metrics.pairwise_distances(
        test_data, train_data), axis=1)

    # Calculate DCR for Synthetic-Train
    dcr_synthetic_train = np.min(metrics.pairwise_distances(
        synthetic_data, train_data), axis=1)

    # Calculate the difference in DCR
    diff_dcr = np.abs(dcr_test_train - dcr_synthetic_train)

    # Calculate Diff DCR in percentage
    diff_dcr_percent = np.mean(diff_dcr / dcr_test_train)  # * 100

    # Calculate Privacy Score
    privacy_score = 1 - diff_dcr_percent

    # Evaluate Privacy Level
    # privacy_level = "High" if diff_dcr_percent < 10 else ("Medium" if diff_dcr_percent < 50 else "Low")

    # Compile results
    results = {
        # "dcr_test_train": dcr_test_train,
        # "dcr_synthetic_train": dcr_synthetic_train,
        # "diff_dcr": diff_dcr,
        "diff_dcr_percent": diff_dcr_percent,
        "privacy_score": privacy_score
        # "Privacy Level": privacy_level
    }

    return results


def compute_nndr_metrics(train_data, test_data, synthetic_data):
    # TODO: revisit
    """
    Calculate Nearest Neighbor Distance Ratio (NNDR) metrics.

    Parameters:
        train_data (pd.DataFrame or pd.Series): Training dataset.
        test_data (pd.DataFrame or pd.Series): Test dataset.
        synthetic_data (pd.DataFrame or pd.Series): Synthetic dataset.

    Returns:
        results (dict): Dictionary containing NNDR metrics and privacy score.
    """
    def nndr(dataset, reference_dataset):
        # Convert pandas data to numpy array if necessary
        if isinstance(dataset, pd.DataFrame) or isinstance(dataset, pd.Series):
            dataset = dataset.values
        if isinstance(reference_dataset, pd.DataFrame) or isinstance(reference_dataset, pd.Series):
            reference_dataset = reference_dataset.values

        # Calculate pairwise distances
        distances = metrics.pairwise_distances(dataset, reference_dataset)
        sorted_distances = np.sort(distances, axis=1)
        closest, second_closest = sorted_distances[:,
                                                   0], sorted_distances[:, 1]
        return closest / second_closest

    nndr_test_train = nndr(test_data, train_data)
    nndr_synthetic_train = nndr(synthetic_data, train_data)

    diff_nndr = np.abs(nndr_test_train - nndr_synthetic_train)
    diff_nndr_percent = np.mean(diff_nndr / nndr_test_train)  # * 100

    privacy_score = 1 - diff_nndr_percent

    # privacy_level = "High" if diff_nndr_percent < 10 else ("Medium" if diff_nndr_percent < 50 else "Low")

    results = {
        "nndr_test_train": nndr_test_train,
        "nndr_synthetic_train": nndr_synthetic_train,
        "diff_nndr": diff_nndr,
        "diff_nndr_percent": diff_nndr_percent,
        "privacy_score": privacy_score
        # "Privacy Level": privacy_level
    }

    return results

# Example usage with pandas DataFrame or Series
# results = compute_nndr_metrics(pd.DataFrame(train_data), pd.DataFrame(test_data), pd.DataFrame(synthetic_data))
# print(results)


# Example Usage
# Assuming train_data, test_data, and synthetic_data are numpy arrays
# results = compute_dcr_metrics(train_data, test_data, synthetic_data)
# print(results)


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
