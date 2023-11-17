# from sdmetrics.single_column import BoundaryAdherence
from sdmetrics.single_column import (CategoryCoverage, MissingValueSimilarity,
                                     RangeCoverage)


def compute_domain_coverage(real_col, synthetic_col, col_data_type):
    """
    Calculate the domain coverage score for a given column.

    Parameters:
        real_col (pandas.Series): Real data column.
        synthetic_col (pandas.Series): Synthetic data column.
        col_data_type (str): Data type of the column.

    Returns:
        domain_coverage_score (float): Coverage score for the domain of the column.
    """
    if col_data_type in ["categorical", "boolean"]:
        # For categorical or boolean columns, use CategoryCoverage metric
        return CategoryCoverage.compute(
            real_data=real_col,
            synthetic_data=synthetic_col
        )
    elif col_data_type in ["numerical", "datetime"]:
        # For numerical or datetime columns, use RangeCoverage metric
        return RangeCoverage.compute(
            real_data=real_col,
            synthetic_data=synthetic_col
        )
    
    return None


def compute_missing_values_coverage(real_col, synthetic_col):
    """
    Calculate the missing values coverage score for a given column.

    Caveat: Missing values are interpreted as NaN.

    Parameters:
        real_col (pandas.Series): Real data column.
        synthetic_col (pandas.Series): Synthetic data column.

    Returns:
        missing_values_score (float): Similarity score for missing values coverage.
    """
    # This metric compares whether the synthetic data has the same
    # proportion of missing values as the real data for a given column.
    # Missing values in your data should be represented as NaN objects.
    
    # Compute missing values coverage score using the MissingValueSimilarity metric
    missing_values_score = MissingValueSimilarity.compute(
        real_data=real_col,
        synthetic_data=synthetic_col
    )
    
    return missing_values_score

# --------------------------------------
# Outlier Coverage
# --------------------------------------


def _calculate_num_outliers(data):
    """
    Calculate the number of outliers in a dataset and the total number of data points.

    Parameters:
        data (pandas.Series): The input data for which to calculate outliers.

    Returns:
        num_outliers (int): Number of outliers in the dataset.
        total_points (int): Total number of data points in the dataset.
    """
    # Calculate the first and third quartiles
    q1, q3 = data.quantile([0.25, 0.75])
    # Calculate the interquartile range (IQR)
    iqr_value = q3 - q1

    # Calculate lower and upper bounds for outliers
    lower_bound = q1 - 1.5 * iqr_value
    upper_bound = q3 + 1.5 * iqr_value

    # Identify outliers within the defined bounds
    outliers = data[(data < lower_bound) | (data > upper_bound)]

    # Calculate the number of outliers and total data points
    num_outliers = len(outliers)
    total_points = len(data)

    return num_outliers, total_points


def compute_outlier_coverage(real_col, synthetic_col):
    """
    Calculate the outlier coverage score between real and synthetic datasets.
    Directions: 
    # Maximize score to ensure synthetic data closely mirrors outlier distribution of real data.
    # Minimize score to reduce emphasis on outliers in synthetic data compared to real data.

    Parameters:
        real_data (pandas.Series): The real dataset containing outliers.
        synthetic_data (pandas.Series): The synthetic dataset to be evaluated.

    Returns:
        score (float): The outlier coverage score.
    """
    # Calculate the number of outliers and total data points for real data
    num_real_outliers, total_real_points = _calculate_num_outliers(real_col)
    # Calculate the number of outliers and total data points for synthetic data
    num_synthetic_outliers, total_synthetic_points = _calculate_num_outliers(
        synthetic_col)

    # Calculate the proportions of outliers
    p_real = num_real_outliers / total_real_points
    p_synthetic = num_synthetic_outliers / total_synthetic_points

    # Calculate the outlier coverage score
    if p_real == 0:
        score = 0 
    else:
        score = min(p_synthetic / p_real, 1)

    return score


# This metrics measures whether a synthetic column respects
# the minimum and maximum values of the real column.
# This metric ignores missing values.
# Types: Numeric and Datetime
# Scores:  0.0 (worst) - 1.0 (best)
# Note: This metric only quantifies cases where the synthetic data is going out of bounds.
# If you're interested in knowing whether the synthetic data covers the
# full range of real values, use the RangeCoverage metric
# BoundaryAdherence.compute(
#     real_data=real_table['column_name'],
#     synthetic_data=synthetic_table['column_name']
# )
