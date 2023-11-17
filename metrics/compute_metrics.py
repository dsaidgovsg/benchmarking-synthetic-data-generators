"""Constants"""
from enum import Enum

# class PrivacyMetrics(Enum):
#     """"""
#     CLASSIFICATION = "classification"
#     REGRESSION = "regression"

from metrics.coverage import compute_domain_coverage, \
    compute_outlier_coverage, \
    compute_missing_values_coverage
from metrics.ml_efficacy import compute_ml_classification, \
    compute_ml_regression
from metrics.privacy import compute_new_row_synthesis, \
    compute_dcr_metrics, \
    compute_nndr_metrics
from metrics.sdv_reports import compute_sdv_quality_report
from metrics.similarity import compute_statistic_similarity, \
    compute_correlation_similarity, \
    compute_jensenshannon_distance, \
    compute_wasserstein_distance

# TODO
"""
Remove the below; for reference only
"""
# univariance -- COL
# "univar_stats_sim": compute_statistic_similarity(real_col, synthetic_col, statistic) -- MAX
# "univar_js_dist": compute_jensenshannon_distance(real_col, synthetic_col, col_data_type) -- MIN
# "univar_wass_dist": compute_wasserstein_distance(real_col, synthetic_col, col_data_type) -- MIN

# coverage -- COL
# "cov_domain": compute_domain_coverage(real_col, synthetic_col, col_data_type) -- MAX
# "cov_misses": compute_missing_values_coverage(real_col, synthetic_col) -- MAX
# "cov_outiers": compute_outlier_coverage(real_col, synthetic_col) -- MAX

# privacy -- DATA
# "priv_new_row_syn": compute_new_row_synthesis(real_data, synthetic_data, metadata,
#   numerical_match_tolerance=0.01,
#   synthetic_sample_percent=None)
# "priv_nndr": compute_nndr_metrics(train_data, test_data, synthetic_data)
# "priv_dcr": compute_dcr_metrics(train_data, test_data, synthetic_data)

# ml_efficacy -- DATA
# "ml_classify": compute_ml_classification(test_data, train_data, target_column, metadata, ml_model)
# "ml_regress": compute_ml_regression(test_data, train_data, target_column, metadata, ml_model)

# sdv_reports (Correlation, Shapes) -- DATA
# "corr_shapes": compute_sdv_quality_report(real_data, synthetic_data, metadata)


def compute_metric(**kw):
    metric_name = kw["metric_name"]

    if metric_name == "univar_stats_sim":
        real_col = kw["real_col"]
        synthetic_col = kw["synthetic_col"]
        statistic = kw["statistic"]
        return compute_statistic_similarity(real_col, synthetic_col, statistic)
    elif metric_name == "univar_js_dist":
        real_col = kw["real_col"]
        synthetic_col = kw["synthetic_col"]
        col_data_type = kw["col_data_type"]
        return compute_jensenshannon_distance(real_col, synthetic_col, col_data_type)
    elif metric_name == "univar_wass_dist":
        real_col = kw["real_col"]
        synthetic_col = kw["synthetic_col"]
        col_data_type = kw["col_data_type"]
        return compute_wasserstein_distance(real_col, synthetic_col, col_data_type)
    elif metric_name == "cov_domain":
        real_col = kw["real_col"]
        synthetic_col = kw["synthetic_col"]
        col_data_type = kw["col_data_type"]
        return compute_domain_coverage(real_col, synthetic_col, col_data_type)
    elif metric_name == "cov_misses":
        real_col = kw["real_col"]
        synthetic_col = kw["synthetic_col"]
        return compute_missing_values_coverage(real_col, synthetic_col)
    elif metric_name == "cov_outiers":
        real_col = kw["real_col"]
        synthetic_col = kw["synthetic_col"]
        return compute_outlier_coverage(real_col, synthetic_col)
    elif metric_name == "priv_new_row_syn":
        real_data = kw["real_data"]
        synthetic_data = kw["synthetic_data"]
        metadata = kw["metadata"]
        numerical_match_tolerance = kw["numerical_match_tolerance"]
        synthetic_sample_percent = kw["synthetic_sample_percent"]
        return compute_new_row_synthesis(real_data,
                                         synthetic_data,
                                         metadata,
                                         numerical_match_tolerance=numerical_match_tolerance,
                                         synthetic_sample_percent=synthetic_sample_percent)
    elif metric_name == "priv_nndr":
        train_data = kw["train_data"]
        test_data = kw["test_data"]
        synthetic_data = kw["synthetic_data"]
        return compute_nndr_metrics(train_data, test_data, synthetic_data)
    elif metric_name == "priv_dcr":
        train_data = kw["train_data"]
        test_data = kw["test_data"]
        synthetic_data = kw["synthetic_data"]
        return compute_dcr_metrics(train_data, test_data, synthetic_data)
    elif metric_name == "ml_classify":
        train_data = kw["train_data"]
        test_data = kw["test_data"]
        target_column = kw["target_column"]
        metadata = kw["metadata"]
        ml_model = kw["ml_model"]
        return compute_ml_classification(
            test_data, train_data, target_column, metadata, ml_model)
    elif metric_name == "ml_regress":
        train_data = kw["train_data"]
        test_data = kw["test_data"]
        target_column = kw["target_column"]
        metadata = kw["metadata"]
        ml_model = kw["ml_model"]
        return compute_ml_regression(test_data, train_data,
                                     target_column, metadata, ml_model)
    elif metric_name == "corr_shapes":
        real_data = kw["real_data"]
        synthetic_data = kw["synthetic_data"]
        metadata = kw["metadata"]
        return compute_sdv_quality_report(real_data, synthetic_data, metadata)
    else:
        raise ValueError("Metric not found.")
