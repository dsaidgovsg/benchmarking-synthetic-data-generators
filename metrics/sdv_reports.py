"Quality and dignostic reports from SDV"

import pandas as pd
# from sdmetrics.reports.single_table import QualityReport
# from sdmetrics.reports.single_table import DiagnosticReport
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic


def get_sdv_quality_report(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict):
    """
    Get a quality report using Synthetic Data Vault (SDV) metrics.

    Parameters:
        real_data (pd.DataFrame): Real dataset.
        synthetic_data (pd.DataFrame): Synthetic dataset.
        metadata (dict): Additional metadata for the evaluation.

    Returns:
        quality_report (dict): Quality report capturing Column Shapes, Column Pair Trends, and more.

    Details:
        Column Shapes
        - numerical, datetime: KSComplement
        - categotrical, boolean: TVComplement
        Column Pair Trends
        - Correlation and Contingency Similarity 
        Ref: https://docs.sdv.dev/sdmetrics/reports/quality-report
        Sample output:
        >> Overall Quality Score: 80.5%
        >> Properties:
        >> - Column Shapes: 82.0%
        >> - Column Pair Trends: 79.0%
    """
    # Generate a quality report using SDV metrics
    report = evaluate_quality(real_data, synthetic_data, metadata)

    # Extract required information from the quality report
    quality_report = {
        "score": report.get_score(),  # Overall quality score
        "properties": report.get_properties(),  # Properties and their scores
        # Column Shapes details
        "column_shapes": report.get_details(property_name='Column Shapes'),
        # Column Pair Trends details
        "column_pair_trends": report.get_details(property_name='Column Pair Trends')
    }

    return quality_report


def get_sdv_diagnostic_report(real_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame,
                              metadata: dict):
    """
    Get a diagnostic report using Synthetic Data Vault (SDV) metrics.

    Parameters:
        real_data (pd.DataFrame): Real dataset.
        synthetic_data (pd.DataFrame): Synthetic dataset.
        metadata (dict): Additional metadata for the evaluation.

    Returns:
        diagnostic_report (dict): Diagnostic report capturing Synthesis, Coverage, and Boundaries.

    Details:
        The quality report captures the: 
        - Synthesis: Is the synthetic data unique or does it copy the real rows?
        - Coverage: Does the synthetic data cover the range of possible values?
        --- Numeric, Datetime: RangeCoverage 
        --- Categorical, Boolean: CategoryCoverage 
        - Boundaries: Does the synthetic data respect the boundaries set by the real data?
        --- BoundaryAdherence metric to numerical columns only
        Ref: https://docs.sdv.dev/sdmetrics/reports/diagnostic-report
    """
    # Generate a diagnostic report using SDV metrics
    report = run_diagnostic(real_data, synthetic_data, metadata)

    # Extract required information from the diagnostic report
    diagnostic_report = {
        "results": report.get_results(),  # Detailed results of the diagnostic evaluation
        "properties": report.get_properties(),  # Properties and their scores
        # Synthesis details
        "synthesis": report.get_details(property_name='Synthesis'),
        # Coverage details
        "coverage": report.get_details(property_name='Coverage'),
        # Boundaries details
        "boundaries": report.get_details(property_name='Boundaries')
    }

    return diagnostic_report
