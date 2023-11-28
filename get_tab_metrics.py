import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import pandas as pd

from commons.static_vals import (ML_CLASSIFICATION_MODELS,
                                 ML_CLASSIFICATION_TASK_DATASETS,
                                 ML_REGRESSION_MODELS,
                                 ML_REGRESSION_TASK_DATASETS,
                                 ML_TASKS_TARGET_CLASS, ROUNDING_VAL,
                                 SIMILARITY_CHECK_STATISTICS)
from commons.utils import (detect_metadata_with_sdv, get_dataset_with_sdv,
                           shuffle_and_split_dataframe,
                           stratified_split_dataframe)
from metrics import coverage, ml_efficacy, privacy, sdv_reports, similarity


def run_metrics(output_path, exp_dataset_name="adult", exp_synthesizer="ctgan", lib="sdv"):
    """
    Run the various metrics on synthetic and real datasets and save the results to a JSON file.

    Parameters:
        output_path (str): The path where the results will be saved.
        exp_dataset_name (str): The name of the dataset being evaluated.
        exp_synthesizer (str): The name of the synthesizer being used.
        lib (str): The library being used for synthesis.
    """
    # Constants
    MODALITY = "tabular"

    # Load synthetic data
    syn_data_path = f"{BASE}/{exp_synthesizer}/{exp_dataset_name}/{exp_dataset_name}_{exp_synthesizer}_synthetic_data.csv"
    # syn_data_path = f"{BASE}/{lib}/{MODALITY}/{exp_synthesizer}/{exp_dataset_name}/{exp_dataset_name}_{exp_synthesizer}_synthetic_data.csv"
    # syn_data_path = f"llm_out_25aug/{exp_dataset_name}/{exp_dataset_name}_synthetic_data.csv" # llm
    synthetic_data = pd.read_csv(syn_data_path)
    if 'Unnamed: 0' in synthetic_data.columns:
        synthetic_data.drop(columns=['Unnamed: 0'], inplace=True)

    # Load real data and metadata
    if exp_dataset_name == "drugs":
        real_dataset = pd.read_csv(
            "sample_datasets/drugs.csv")
        metadata_class = detect_metadata_with_sdv(real_dataset)
    elif exp_dataset_name == "health_insurance":
        real_dataset = pd.read_csv("sample_datasets/health_insurance.csv")
        metadata_class = detect_metadata_with_sdv(real_dataset)
    elif exp_dataset_name == "loan":
        real_dataset = pd.read_csv("sample_datasets/loan.csv")
        metadata_class = detect_metadata_with_sdv(real_dataset)
    else:
        real_dataset, metadata_class = get_dataset_with_sdv(
            "single_table", exp_dataset_name)

    metadata_dict = metadata_class.to_dict()

    # Split real data
    if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
        (X_train, y_train), (X_test, y_test) = stratified_split_dataframe(
            real_dataset, ML_TASKS_TARGET_CLASS[exp_dataset_name], False)
        real_data_train = pd.concat([X_train, y_train], axis=1)
        real_data_test = pd.concat([X_test, y_test], axis=1)
    else:
        (real_data_train, real_data_test) = shuffle_and_split_dataframe(
            real_dataset, False)

    col_md = metadata_dict["columns"]

    # Initialize results dictionary
    results = {
        "privacy": {},
        "coverage": {},
        "ml_efficacy": {},
        "similarity": {},
        "sdv_quality_report": {}
    }

    begin_compute_time = time.time()

    # ------------------
    # Privacy Metrics
    # ------------------
    # try:
    #     begin_time = time.time()
    #     new_row_synthesis = privacy.compute_new_row_synthesis(
    #         real_dataset, synthetic_data, metadata_dict)
    #     results["privacy"]["new_row_synthesis"] = round(
    #         new_row_synthesis, ROUNDING_VAL)
    #     LOGGER.info(f"SUCCESS: new_row_synthesis: {new_row_synthesis}")
    #     results["privacy"]["timing"] = time.time() - begin_time
    # except Exception as e:
    #     LOGGER.error(e)

    # ------------------
    # Coverage Metrics
    # ------------------
    try:
        # Loop over columns
        results["coverage"]["domain_coverage"] = {}
        results["coverage"]["missing_values_coverage"] = {}
        results["coverage"]["outlier_coverage"] = {}
        begin_time = time.time()
        for k, v in col_md.items():
            real_col = real_data_test[k]
            synthetic_col = synthetic_data[k]
            col_type = v["sdtype"]

            try:
                domain_coverage = coverage.compute_domain_coverage(
                    real_col, synthetic_col, col_type)
                results["coverage"]["domain_coverage"][k] = round(
                    domain_coverage, ROUNDING_VAL)
            except Exception as e:
                LOGGER.error(f"compute_domain_coverage error: {e}")

            try:
                missing_values_coverage = coverage.compute_missing_values_coverage(
                    real_col, synthetic_col)
                results["coverage"]["missing_values_coverage"][k] = round(
                    missing_values_coverage, ROUNDING_VAL)
            except Exception as e:
                LOGGER.error(f"compute_missing_values_coverageerror: {e}")

            try:
                if col_type == "numerical":
                    outlier_coverage = coverage.compute_outlier_coverage(
                        real_col, synthetic_col)
                    results["coverage"]["outlier_coverage"][k] = outlier_coverage
            except Exception as e:
                LOGGER.error(f"compute_outlier_coverage error: {e}")
        results["coverage"]["timing"] = time.time() - begin_time
    except Exception as e:
        LOGGER.error(e)

    # ------------------
    # Similarity Metrics
    # ------------------
    try:
        cat_cols = []
        results["similarity"]["statistic"] = {}
        results["similarity"]["distance"] = {}

        begin_time = time.time()
        # Loop over columns
        for k, v in col_md.items():
            real_col = real_data_test[k]
            synthetic_col = synthetic_data[k]
            col_type = v["sdtype"]

            try:
                # Statistic similarity
                if col_type == "numerical":
                    results["similarity"]["statistic"][k] = {}
                    for stat in SIMILARITY_CHECK_STATISTICS:
                        statistic_similarity = similarity.compute_statistic_similarity(
                            real_col, synthetic_col, stat)
                        results["similarity"]["statistic"][k][stat] = round(
                            statistic_similarity, ROUNDING_VAL)

                        # Distance similarity
                        distance = similarity.compute_distance(
                            real_col, synthetic_col, col_type)
                        results["similarity"]["distance"][k] = distance

                if col_type == "categorical":
                    cat_cols.append(k)
            except Exception as e:
                LOGGER.error(f"Statistic similarity error: {e}")

        try:
            # ------------------
            # Correlation similarity
            # ------------------
            correlation_similarity, _, _ = similarity.compute_correlation_similarity(
                real_data_test, synthetic_data, {"categorical": cat_cols})
            results["similarity"]["correlation"] = round(
                correlation_similarity, ROUNDING_VAL)

            # fig = plt.figure()
            # fig.add_axes(ax_real_corr)
            # # fig = ax_real_corr.figure

        except Exception as e:
            LOGGER.error(f"Correlation similarity error: {e}")
        results["similarity"]["timing"] = time.time() - begin_time
    except Exception as e:
        LOGGER.error(e)

    # ------------------
    # ML Efficacy Metrics
    # ------------------
    try:
        results["ml_efficacy"] = {}
        begin_time = time.time()
        if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
            for ml_model in ML_CLASSIFICATION_MODELS:
                f1_real = ml_efficacy.compute_ml_classification(
                    real_data_test, real_data_train, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                f1_synthetic = ml_efficacy.compute_ml_classification(
                    real_data_test, synthetic_data, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                results["ml_efficacy"][f"{ml_model}_classification"] = {
                    "synthetic_f1": round(f1_synthetic, ROUNDING_VAL),
                    "real_f1": round(f1_real, ROUNDING_VAL)}
        elif exp_dataset_name in ML_REGRESSION_TASK_DATASETS:
            for ml_model in ML_REGRESSION_MODELS:
                r2_real = ml_efficacy.compute_ml_regression(
                    real_data_test, real_data_train, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                r2_synthetic = ml_efficacy.compute_ml_regression(
                    real_data_test, synthetic_data, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                results["ml_efficacy"][f"{ml_model}_regression"] = {
                    "synthetic_r2": round(r2_synthetic, ROUNDING_VAL),
                    "real_r2": round(r2_real, ROUNDING_VAL)}
        results["ml_efficacy"]["timing"] = time.time() - begin_time
    except Exception as e:
        LOGGER.error(e)

    # --------------------------
    # SDV Quality Report Metrics
    # --------------------------
    try:
        begin_time = time.time()
        q_report = sdv_reports.compute_sdv_quality_report(
            real_data_test, synthetic_data, metadata_class)
        results["sdv_quality_report"]["score"] = q_report.get_score()

        # Process and store distribution scores
        q_report_cols_df = q_report.get_details(property_name='Column Shapes')
        dis_dict = {}
        for _, row in q_report_cols_df.iterrows():
            column = row['Column']
            metric = row['Metric']
            quality_score = row['Quality Score']
            dis_dict[column] = {
                "metric": metric,
                "score": quality_score
            }
        results["sdv_quality_report"]["distribution"] = dis_dict

        # Store quality report details
        q_report_corr_df = q_report.get_details(
            property_name='Column Pair Trends')
        q_report_corr_df.to_csv(
            f"{output_path}/{exp_dataset_name}_{exp_synthesizer}_correlation.csv", index=False)

        results["sdv_quality_report"]["timing"] = time.time() - begin_time
        LOGGER.info(f"q_report: {results['sdv_quality_report']['score']}")
        LOGGER.info(f"q_report: {q_report.get_properties()}")
    except Exception as e:
        LOGGER.error(e)

    results["total_time"] = time.time() - begin_compute_time

    # Save execution data to JSON file
    with open(f"{output_path}/{exp_dataset_name}_{exp_synthesizer}_metrics.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    # BASE = "az_outputs_23aug/2023-08-20"
    BASE = "final_outs/final_sdv_tabular_23aug"
    LIB = "sdv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_set", "--ds", type=str, default="s3",
                        help="enter set of data. \
                        Possible values - {s1, s2, s3}")
    parser.add_argument("--synthesizer", "--s", type=str, default="ctgan",
                        help="enter synthesizer name \
                            Possible values - {ctgan, tvae, gaussian_copula}")
    parser.add_argument("--output_folder", "--o", type=str, default="outputs")

    args = parser.parse_args()

    exp_data_set_name: str = args.data_set
    exp_synthesizer: str = args.synthesizer
    output_folder: str = args.output_folder

    # exp_synthesizer = "ctgan"  # gaussian_copula, tvae
    BASE_OUTPUT_PATH = f"{output_folder}/{exp_synthesizer}"

    # if not os.path.exists(BASE_OUTPUT_PATH):
    #     os.makedirs(BASE_OUTPUT_PATH)

    # temp naming
    if exp_data_set_name == "s1":
        exp_data_set = ["adult", "drugs", "intrusion"]
    elif exp_data_set_name == "s2":
        exp_data_set = ["loan", "covtype", "child"]
    elif exp_data_set_name == "llm":
        exp_data_set = ["adult"] #"health_insurance"]
    else:
        exp_data_set = ["health_insurance", "census", "credit"]

    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)

    logging.basicConfig(filename=f"{BASE_OUTPUT_PATH}/{exp_synthesizer}_{exp_data_set_name}.log",
                        format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s ",
                        datefmt="%Y-%m-%d:%H:%M:%S",
                        filemode="w")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)

    for dataset_name in exp_data_set:
        OUTPUT_PATH = BASE_OUTPUT_PATH + "/" + dataset_name
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        LOGGER.info("*"*30)
        LOGGER.info(f"Running for {dataset_name} {exp_synthesizer}")
        LOGGER.info("*"*30)

        try:
            run_metrics(output_path=OUTPUT_PATH,
                        exp_dataset_name=dataset_name,
                        exp_synthesizer=exp_synthesizer,
                        lib=LIB)
        except Exception as e:
            print("hererrereerr", e)
            LOGGER.error(e)
        LOGGER.info("*"*30)
        LOGGER.info(
            f"SUCCESS: Metrics generated for {dataset_name} {exp_synthesizer} ")
        LOGGER.info("*"*30)
