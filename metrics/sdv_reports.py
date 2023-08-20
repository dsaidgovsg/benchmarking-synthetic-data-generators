"Quality and dignostic reports from SDV"

import pandas as pd
# from sdmetrics.reports.single_table import QualityReport
from sdmetrics.reports.single_table import DiagnosticReport
from sdv.evaluation.single_table import evaluate_quality, run_diagnostic


def get_sdv_quality_report(
        real_data: pd.DataFrame,
        synthetic_data: pd.DataFrame,
        metadata: dict):
    """ 
        The quality report captures the Column Shapes, Column Pair Trends and Cardinality.
        Ref: https://docs.sdv.dev/sdmetrics/reports/quality-report
    """
    report = evaluate_quality(real_data, synthetic_data, metadata)
    # score = report.get_score()
    # properties = report.get_properties()
    # column_shapes = report.get_details(property_name='Column Shapes')
    # column_pair_trends = report.get_details(property_name='Column Pair Trends')
    return {
        "score": report.get_score(),
        # "properties": report.get_properties(),  # DF
        # # DF
        # "column_shapes": report.get_details(property_name='Column Shapes'),
        # # DF
        # "column_pair_trends": report.get_details(property_name='Column Pair Trends')
    }


def get_sdv_diagnostic_report(real_data: pd.DataFrame,
                              synthetic_data: pd.DataFrame,
                              metadata: dict):
    """ 
        The quality report captures the: 
        - Synthesis: Is the synthetic data unique or does it copy the real rows?
        - Coverage: Does the synthetic data cover the range of possible values?
        - Boundaries: Does the synthetic data respect the boundaries set by the real data?
        Ref: https://docs.sdv.dev/sdmetrics/reports/diagnostic-report
    """
    report = run_diagnostic(real_data, synthetic_data, metadata)
    # results = report.get_results()
    # properties = report.get_properties()
    # synthesis = report.get_details(property_name='Synthesis')
    # coverage = report.get_details(property_name='Coverage')
    # boundaries = report.get_details(property_name='Boundaries')
    return {
        "results": report.get_results(),
        "properties":  report.get_properties(),
        # "synthesis": report.get_details(property_name='Synthesis'),  # DF
        # "coverage": report.get_details(property_name='Coverage'),  # DF
        # "boundaries": report.get_details(property_name='Boundaries')  # DF
    }
