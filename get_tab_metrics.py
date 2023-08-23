"Compute metrics"
import os
import time
import json
import pandas as pd
import logging
from metrics import coverage, privacy, ml_efficacy, similarity, sdv_reports

from commons.static_vals import SIMILARITY_CHECK_STATISTICS, \
    ML_CLASSIFICATION_MODELS, \
    ML_REGRESSION_MODELS, \
    ML_CLASSIFICATION_TASK_DATASETS, \
    ML_REGRESSION_TASK_DATASETS, \
    ML_TASKS_TARGET_CLASS, \
    ROUNDING_VAL
from commons.utils import detect_metadata_with_sdv, \
    get_dataset_with_sdv, \
    shuffle_and_split_dataframe, \
    stratified_split_dataframe


def run_metrics(output_path, exp_dataset_name="adult", exp_synthesizer="ctgan", lib="sdv"):
    MODALITY = "tabular"

    # -----------------------
    # Load synthetic data
    # -----------------------
    syn_data_path = f"{BASE}/{lib}/{MODALITY}/{exp_synthesizer}/{exp_dataset_name}/{exp_dataset_name}_{exp_synthesizer}_synthetic_data.csv"
    synthetic_data = pd.read_csv(syn_data_path)
    if 'Unnamed: 0' in synthetic_data.columns:
        synthetic_data.drop(columns=['Unnamed: 0'], inplace=True)

    if exp_dataset_name == "drugs":
        real_dataset = pd.read_csv(
            f"sample_datasets/accidential_drug_deaths.csv")
        metadata_class = detect_metadata_with_sdv(real_dataset)
    elif exp_dataset_name == "health_insurance":
        real_dataset = pd.read_csv(
            f"sample_datasets/health_insurance.csv")
        metadata_class = detect_metadata_with_sdv(real_dataset)
    elif exp_dataset_name == "loan":
        real_dataset = pd.read_csv(
            f"sample_datasets/personal_loan.csv")
        metadata_class = detect_metadata_with_sdv(real_dataset)
    else:
        real_dataset, metadata_class = get_dataset_with_sdv(
            "single_table", exp_dataset_name)

    metadata_dict = metadata_class.to_dict()

    # -----------------------------------
    # Load real data and split
    # -----------------------------------
    if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
        (X_train, y_train), (X_test, y_test) = stratified_split_dataframe(real_dataset,
                                                                          ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                                                          False)
        # Merge X_train and y_train columns horizontally
        real_data_train = pd.concat([X_train, y_train], axis=1)
        real_data_test = pd.concat([X_test, y_test], axis=1)
    else:
        (real_data_train, real_data_test) = shuffle_and_split_dataframe(
            real_dataset, False)

    col_md = metadata_dict["columns"]

    # TODO: track time taken to generate the metrics

    results = {
        "privacy": {},
        "coverage": {},
        "ml_efficacy": {},
        "similarity": {},
        "sdv_quality_report": {}
    }

    # timings = {
    #     "privacy": {},
    #     "coverage": {},
    #     "ml_efficacy": {},
    #     "similarity": {},
    #     "quality_report": {}
    # }

    begin_compute_time = time.time()
    # ------------------
    # Privacy
    # ------------------
    # TODO: uncomment
    try:
        begin_time = time.time()
        new_row_synthesis = privacy.compute_new_row_synthesis(
            real_dataset, synthetic_data, metadata_dict)

        results["privacy"]["new_row_synthesis"] = round(
            new_row_synthesis, ROUNDING_VAL)

        # timings["privacy"]["new_row_synthesis"] = time.time() - begin_time

        LOGGER.info("SUCCESS: new_row_synthesis: ", new_row_synthesis)
    except Exception as e:
        LOGGER.error(e)

    # TODO: Failed for non-numeric types
    # distance_to_closest_records = privacy.compute_distance_to_closest_records(
    #     real_data, synthetic_data)
    # print("distance_to_closest_records: ", distance_to_closest_records)
    # breakpoint()

    cat_cols = []

    results["coverage"]["domain_coverage"] = {}
    results["coverage"]["missing_values_coverage"] = {}
    results["coverage"]["outlier_coverage"] = {}
    results["similarity"]["statistic"] = {}
    results["similarity"]["distance"] = {}
    for k, v in col_md.items():

        real_col = real_data_test[k]
        synthetic_col = synthetic_data[k]
        col_type = v["sdtype"]

        # ------------------
        # Coverage
        # ------------------
        try:
            domain_coverage = coverage.compute_domain_coverage(
                real_col, synthetic_col, col_type)
            LOGGER.info("domain_coverage: ", domain_coverage)
            results["coverage"]["domain_coverage"][k] = round(
                domain_coverage, ROUNDING_VAL)
            missing_values_coverage = coverage.compute_missing_values_coverage(
                real_col, synthetic_col)
            LOGGER.info("missing_values_coverage_score: ",
                        missing_values_coverage)
            results["coverage"]["missing_values_coverage"][k] = round(
                missing_values_coverage, ROUNDING_VAL)
        except Exception as e:
            LOGGER.error(e)

        try:
            if col_type == "numerical":
                outlier_coverage = coverage.compute_outlier_coverage(
                    real_col, synthetic_col)
                LOGGER.info("outlier_coverage: ", outlier_coverage)
                results["coverage"]["outlier_coverage"][k] = outlier_coverage
        except Exception as e:
            LOGGER.error(e)

        if col_type == "categorical":
            cat_cols.append(k)

        # ------------------
        # Similarity
        # ------------------
        try:
            results["similarity"]["statistic"][k] = {}
            for stat in SIMILARITY_CHECK_STATISTICS:
                statistic_similarity = similarity.compute_statistic_similarity(
                    real_col, synthetic_col, stat)
                LOGGER.info("statistic_similarit: ", statistic_similarity)
                results["similarity"]["statistic"][k][stat] = round(
                    statistic_similarity, ROUNDING_VAL)
        except Exception as e:
            LOGGER.info(e)

        try:
            distance = similarity.compute_distance(
                real_col, synthetic_col, col_type)
            LOGGER.info("distance: ", distance)
            results["similarity"]["distance"][k] = distance
        except Exception as e:
            LOGGER.error(e)

    try:
        correlation_similarity = similarity.compute_correlation_similarity(
            real_data_test, synthetic_data, {"categorical": cat_cols})

        LOGGER.info("correlation_similarity: ", correlation_similarity)
        results["similarity"]["correlation"] = round(
            correlation_similarity, ROUNDING_VAL)
    except Exception as e:
        LOGGER.error(e)

    # ------------------
    # ML Efficacy
    # ------------------
    if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
        try:
            results["ml_efficacy"] = {}
            for ml_model in ML_CLASSIFICATION_MODELS:
                LOGGER.info("training on: ", ml_model)
                f1_real = ml_efficacy.compute_ml_classification(
                    real_data_test, real_data_train, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                LOGGER.info("f1_real: ", f1_real)
                f1_synthetic = ml_efficacy.compute_ml_classification(
                    real_data_test, synthetic_data, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                LOGGER.info("f1_synthetic: ", f1_synthetic)

                results["ml_efficacy"][f"{ml_model}_classification"] = {
                    "synthetic_f1": round(f1_synthetic, ROUNDING_VAL),
                    "real_f1": round(f1_real, ROUNDING_VAL)}
        except Exception as e:
            LOGGER.error(e)

    if exp_dataset_name in ML_REGRESSION_TASK_DATASETS:
        try:
            for ml_model in ML_REGRESSION_MODELS:
                r2_real = ml_efficacy.compute_ml_regression(
                    real_data_test, real_data_train, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                LOGGER.info("r2_real: ", r2_real)
                r2_synthetic = ml_efficacy.compute_ml_regression(
                    real_data_test, synthetic_data, ML_TASKS_TARGET_CLASS[exp_dataset_name], metadata_dict, ml_model)
                LOGGER.info("r2_synthetic: ", r2_synthetic)

                results["ml_efficacy"][f"{ml_model}_regression"] = {
                    "synthetic_r2": round(r2_synthetic, ROUNDING_VAL),
                    "real_r2": round(r2_real, ROUNDING_VAL)}
        except Exception as e:
            LOGGER.error(e)

    # ------------------
    # SDV Quality Report
    # ------------------
    try:
        q_report = sdv_reports.compute_sdv_quality_report(
            real_data_test, synthetic_data, metadata_class)

        results["sdv_quality_report"]["score"] = q_report.get_score()
        dist_scores = list(q_report.get_properties()["Score"])
        results["sdv_quality_report"]["column_shapes"] = dist_scores[0]
        results["sdv_quality_report"]["column_pair_trends"] = dist_scores[1]

        q_report_cols_df = q_report.get_details(property_name='Column Shapes')
       # Iterate over rows and store in a dictionary
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

        q_report_corr_df = q_report.get_details(
            property_name='Column Pair Trends')
        q_report_corr_df.to_csv(
            f"{output_path}/{exp_dataset_name}_{exp_synthesizer}_synthetic_data.csv", index=False)

        LOGGER.info("q_report: ", q_report.get_score())
        LOGGER.info("q_report: ", q_report.get_properties())
        # print("q_report: ", q_report.get_details(
        #     property_name='Column Shapes'))
        # print("q_report: ", q_report.get_details(
        #     property_name='Column Pair Trends'))
    except Exception as e:
        LOGGER.error(e)

    results["total_time"] = time.time() - begin_compute_time

    # save execution data
    with open(f"{output_path}/{exp_dataset_name}_{exp_synthesizer}_metrics.json", "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":

    # TODO: rename the BASE
    BASE = "az_outputs_23aug/2023-08-20"
    # BASE = "az_synth_23aug/2023-08-20"

    LIB = "sdv"
    EXP_SYNTHESIZER = "ctgan"
    OUTPUT_PATH = f"metrics_output/{EXP_SYNTHESIZER}"

    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    for dataset_name in ["adult"]:

        OUTPUT_PATH = OUTPUT_PATH + "/" + dataset_name
        if not os.path.exists(OUTPUT_PATH):
            os.makedirs(OUTPUT_PATH)

        logging.basicConfig(filename=f"{OUTPUT_PATH}/{EXP_SYNTHESIZER}_{dataset_name}.log",
                            format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s ",
                            datefmt="%Y-%m-%d:%H:%M:%S",
                            filemode="w")
        LOGGER = logging.getLogger(__name__)
        LOGGER.setLevel(logging.DEBUG)

        try:
            run_metrics(output_path=OUTPUT_PATH,
                        exp_dataset_name=dataset_name,
                        exp_synthesizer=EXP_SYNTHESIZER,
                        lib=LIB)
        except Exception as e:
            LOGGER.error(e)
