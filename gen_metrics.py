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
# from metrics import coverage, ml_efficacy, privacy, sdv_reports, similarity
from metrics.compute_metrics import compute_metric


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

    if 'Unnamed: 0.1' in synthetic_data.columns:
        synthetic_data.drop(columns=['Unnamed: 0.1'], inplace=True) 

    # Load real data and metadata
    # if exp_dataset_name == "drugs":
    #     real_dataset = pd.read_csv(
    #         "sample_datasets/drugs.csv")
    #     metadata_class = detect_metadata_with_sdv(real_dataset)
    # elif exp_dataset_name == "health_insurance":
    #     real_dataset = pd.read_csv("sample_datasets/health_insurance.csv")
    #     metadata_class = detect_metadata_with_sdv(real_dataset)
    # elif exp_dataset_name == "loan":
    # /Users/anshusingh/DPPCC/whitespace/benchmarking-synthetic-data-generators/

    if exp_dataset_name in ["taxi", "nasdaq"]:
        real_dataset = pd.read_csv(f"all_sample_datasets/sequential/{exp_dataset_name}.csv")
    else:
        real_dataset = pd.read_csv(f"all_sample_datasets/{exp_dataset_name}.csv")

    if 'Unnamed: 0' in real_dataset.columns:
        real_dataset.drop(columns=['Unnamed: 0'], inplace=True)

    # for llm :(
    if 'Unnamed: 0.1' in real_dataset.columns:
        real_dataset.drop(columns=['Unnamed: 0.1'], inplace=True) 

    if exp_dataset_name in ["loan", "drugs"]:
        # Check if the column exists in DataFrame
        if "ID" in real_dataset.columns:
            real_dataset.drop(columns=["ID"], inplace=True)
        if "ID" in synthetic_data.columns:
            synthetic_data.drop(columns=["ID"], inplace=True)

    metadata_class = detect_metadata_with_sdv(real_dataset)
    # else:
    #     real_dataset, metadata_class = get_dataset_with_sdv(
    #         "single_table", exp_dataset_name)

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

    # print('Holand-Netherlands' in real_data_test['native-country'].values)
    # print('Holand-Netherlands' in real_data_train['native-country'].values)
    # breakpoint()

    print(synthetic_data.shape, real_data_test.shape)

    if len(synthetic_data) > len(real_data_test):
        synthetic_data_test = synthetic_data.sample(
            len(real_data_test), random_state=32)
    else:
        real_data_test = real_data_test.sample(
            len(synthetic_data), random_state=32)
        synthetic_data_test = synthetic_data

    # synthetic_data_test = syn

    print(synthetic_data.shape, real_data_test.shape, synthetic_data_test.shape)

    col_md = metadata_dict["columns"]

    # Initialize results dictionary
    results = {
        "shape": {
            "synthetic_data": synthetic_data.shape,
            "real_data": real_dataset.shape,
            "real_data_train": real_data_train.shape,
            "real_data_test": real_data_test.shape,
            "synthetic_data_test": synthetic_data_test.shape
        },
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
    try:
        results["privacy"]["new_row_synthesis"] = {}
        # results["privacy"]["new_row_synthesis"] = {}
        # results["privacy"]["distance_to_closest_record"] = {}
        # results["privacy"]["nearest_neighbor_distance_ratio"] = {}

        begin_time = time.time()

        # TODO: add the below metrics later; need to convert to numerical
        # print("train_data :", real_data_train.shape,
        #     "test_data: ", real_data_test.shape,
        #     "synthetic_data: ", synthetic_data.shape)
        # priv_dcr = compute_metric(
        #     metric_name="priv_dcr",
        #     train_data=real_data_train,
        #     test_data=real_data_test,
        #     synthetic_data=synthetic_data.sample(n=len(real_data_test)),
        #     metadata=metadata_dict)
        # print("priv_dcr --->", priv_dcr)
        # results["privacy"]["distance_to_closest_record"] = priv_dcr

        # priv_nndr = compute_metric(
        #     metric_name="priv_nndr",
        #     train_data=real_data_train,
        #     test_data=real_data_test,
        #     synthetic_data=synthetic_data,
        #     metadata=metadata_dict)

        # results["privacy"]["nearest_neighbor_distance_ratio"] = priv_nndr
        # print("priv_nndr --->", priv_nndr)

        # results["privacy"]["new_row_synthesis"] = round(
        #     new_row_synthesis, ROUNDING_VAL)
        # LOGGER.info(f"SUCCESS: new_row_synthesis: {new_row_synthesis}")
        # results["privacy"]["timing"] = time.time() - begin_time

        # begin_time = time.time()
        print(real_data_train.columns)
        print("~"*10)
        print(synthetic_data_test.columns)

        synthetic_sample_percent = 0.1
        numerical_match_tolerance = 0.01
        new_row_synthesis = compute_metric(
            metric_name="priv_new_row_syn",
            real_data=real_data_train,
            synthetic_data=synthetic_data_test,
            metadata=metadata_dict,
            numerical_match_tolerance=0.01,
            synthetic_sample_percent=synthetic_sample_percent)
        # print("new_row_synthesis-->", new_row_synthesis)

        results["privacy"]["new_row_synthesis"]["numerical_match_tolerance"] = numerical_match_tolerance
        results["privacy"]["new_row_synthesis"]["synthetic_sample_percent"] = synthetic_sample_percent
        results["privacy"]["new_row_synthesis"]["score"] = round(
            new_row_synthesis, ROUNDING_VAL)
        LOGGER.info(f"SUCCESS: new_row_synthesis: {new_row_synthesis}")

        results["privacy"]["timing"] = time.time() - begin_time
        print(f"SUCCESS: new_row_synthesis: {new_row_synthesis}")
    except Exception as e:
        LOGGER.error(e)
        print(f"FAILURE: new_row_synthesis: {e}")

    # ------------------
    # Coverage Metrics
    # ------------------
    try:
        # Loop over columns
        results["coverage"]["domain_coverage"] = {}
        results["coverage"]["missing_values_coverage"] = {}
        results["coverage"]["outlier_coverage"] = {}
        begin_time = time.time()
        print("RUNNING coverage metrics")
        for k, v in col_md.items():
            real_col = real_data_test[k]
            synthetic_col = synthetic_data_test[k]
            col_data_type = v["sdtype"]

            try:
                domain_coverage = compute_metric(metric_name="cov_domain",
                                                 real_col=real_col,
                                                 synthetic_col=synthetic_col,
                                                 col_data_type=col_data_type)
                results["coverage"]["domain_coverage"][k] = round(
                    domain_coverage, ROUNDING_VAL)
                # print(f"SUCCESS: domain_coverage: {domain_coverage}")
            except Exception as e:
                print(e)
                LOGGER.error(f"compute_domain_coverage error: {e}")
                # print(f"FAILURE: domain_coverage: {e}")

            try:
                missing_values_coverage = compute_metric(metric_name="cov_misses",
                                                         real_col=real_col,
                                                         synthetic_col=synthetic_col)
                results["coverage"]["missing_values_coverage"][k] = round(
                    missing_values_coverage, ROUNDING_VAL)
                # print(f"SUCCESS: cov_misses: {e}")
            except Exception as e:
                print(e)
                LOGGER.error(f"compute_missing_values_coverageerror: {e}")

            try:
                if col_data_type == "numerical":
                    outlier_coverage = compute_metric(metric_name="cov_outiers",
                                                      real_col=real_col,
                                                      synthetic_col=synthetic_col)
                    results["coverage"]["outlier_coverage"][k] = outlier_coverage
            except Exception as e:
                print(e)
                LOGGER.error(f"compute_outlier_coverage error: {e}")
        results["coverage"]["timing"] = time.time() - begin_time
    except Exception as e:
        print(e)
        LOGGER.error(e)

    # ------------------
    # Similarity Metrics
    # ------------------
    try:
        cat_cols = []
        results["similarity"]["statistic"] = {}
        results["similarity"]["wass_distance"] = {}
        results["similarity"]["js_distance"] = {}

        begin_time = time.time()
        print("RUNNING similarity metrics")
        # Loop over columns
        for k, v in col_md.items():
            real_col = real_data_test[k]
            synthetic_col = synthetic_data_test[k]
            col_data_type = v["sdtype"]

            try:
                # Statistic similarity
                if col_data_type == "numerical":
                    results["similarity"]["statistic"][k] = {}
                    for stat in SIMILARITY_CHECK_STATISTICS:
                        statistic_similarity = compute_metric(
                            metric_name="univar_stats_sim",
                            real_col=real_col,
                            synthetic_col=synthetic_col,
                            statistic=stat)
                        results["similarity"]["statistic"][k][stat] = round(
                            statistic_similarity, ROUNDING_VAL)

                        # Wasserstein Distance
                        distance = compute_metric(metric_name="univar_wass_dist",
                                                  real_col=real_col,
                                                  synthetic_col=synthetic_col,
                                                  col_data_type=col_data_type)
                        results["similarity"]["wass_distance"][k] = distance

                        # Jensenshannon Distance
                        distance = compute_metric(metric_name="univar_js_dist",
                                                  real_col=real_col,
                                                  synthetic_col=synthetic_col,
                                                  col_data_type=col_data_type)
                        results["similarity"]["js_distance"][k] = distance

                if col_data_type == "categorical":
                    cat_cols.append(k)
            except Exception as e:
                print(e)
                LOGGER.error(f"Statistic similarity error: {e}")
        results["similarity"]["timing"] = time.time() - begin_time
    except Exception as e:
        print(e)
        LOGGER.error(e)

        #     try:
    #         # ------------------
    #         # Correlation similarity
    #         # ------------------
    #         correlation_similarity, _, _ = compute_metric(
    #             metric_name="",
    #             real_data_test=real_data_test,
    #             synthetic_data=synthetic_data_test,
    #             {"categorical": cat_cols})
    #         results["similarity"]["correlation"] = round(
    #             correlation_similarity, ROUNDING_VAL)

    #         # fig = plt.figure()
    #         # fig.add_axes(ax_real_corr)
    #         # # fig = ax_real_corr.figure

    #     except Exception as e:
    #         print(e)
    #         LOGGER.error(f"Correlation similarity error: {e}")

    # --------------------------
    # SDV Quality Report Metrics
    # --------------------------
    try:
        print("RUNNING Quality report metrics")
        begin_time = time.time()
        q_report = compute_metric(metric_name="corr_shapes",
                                  real_data=real_data_test,
                                  synthetic_data=synthetic_data_test,
                                  metadata=metadata_class)
        results["sdv_quality_report"]["score"] = q_report.get_score()

        # Process and store distribution scores
        q_report_cols_df = q_report.get_details(property_name='Column Shapes')
        dis_dict = {}

        print("_"*20)
        print(q_report_cols_df)
        print("_"*20)
        for _, row in q_report_cols_df.iterrows():
            column = row['Column']
            metric = row['Metric']
            quality_score = row['Score']
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
        print(e)
        LOGGER.error(e)
    print("DONE WITH THE QUALITY REPORT!-----")

    # ------------------
    # ML Efficacy Metrics
    # ------------------
    try:
        results["ml_efficacy"] = {}

        if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS or exp_dataset_name in ML_REGRESSION_TASK_DATASETS:
            # categorical_columns = real_data_test.select_dtypes(include=['category', 'object']).columns
            for k, v in col_md.items():
                if v["sdtype"] != "categorical":
                    continue

                additional_categories_real = list(
                    set(real_data_test[k].unique()) - set(real_data_train[k].unique()))
                additional_categories_syn = list(
                    set(real_data_test[k].unique()) - set(synthetic_data_test[k].unique()))

                # Drop rows in real_data_test that contain any of these additional categories
                if additional_categories_real:
                    real_data_test = real_data_test[~real_data_test[k].isin(
                        additional_categories_real)]

                if additional_categories_syn:
                    real_data_test = real_data_test[~real_data_test[k].isin(
                        additional_categories_syn)]

        if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
            begin_time = time.time()
            print("RUNNING machine learning metrics")
            for ml_model in ML_CLASSIFICATION_MODELS:
                try:
                    f1_real = compute_metric(metric_name="ml_classify",
                                             train_data=real_data_train,
                                             test_data=real_data_test,
                                             target_column=ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                             metadata=metadata_dict,
                                             ml_model=ml_model)
                    f1_synthetic = compute_metric(metric_name="ml_classify",
                                                  train_data=synthetic_data_test,
                                                  test_data=real_data_test,
                                                  target_column=ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                                  metadata=metadata_dict,
                                                  ml_model=ml_model)
                    results["ml_efficacy"][f"{ml_model}_classification"] = {
                        "synthetic_f1": round(f1_synthetic, ROUNDING_VAL),
                        "real_f1": round(f1_real, ROUNDING_VAL)}
                except Exception as e:
                    print(e)
        elif exp_dataset_name in ML_REGRESSION_TASK_DATASETS:
            for ml_model in ML_REGRESSION_MODELS:
                try:
                    r2_real = compute_metric(metric_name="ml_regress",
                                             train_data=real_data_train,
                                             test_data=real_data_test,
                                             target_column=ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                             metadata=metadata_dict,
                                             ml_model=ml_model)
                    r2_synthetic = compute_metric(metric_name="ml_regress",
                                                  train_data=synthetic_data_test,
                                                  test_data=real_data_test,
                                                  target_column=ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                                  metadata=metadata_dict,
                                                  ml_model=ml_model)
                    results["ml_efficacy"][f"{ml_model}_regression"] = {
                        "synthetic_r2": round(r2_synthetic, ROUNDING_VAL),
                        "real_r2": round(r2_real, ROUNDING_VAL)}
                except Exception as e:
                    print(e)

        results["ml_efficacy"]["timing"] = time.time() - begin_time
    except Exception as e:
        print(e)
        LOGGER.error(e)

    results["total_time"] = time.time() - begin_compute_time

    print("~"*20)
    print(f"{output_path}/{exp_dataset_name}_{exp_synthesizer}_metrics.json")

    # Save execution data to JSON file
    with open(f"{output_path}/{exp_dataset_name}_{exp_synthesizer}_metrics.json", "w") as json_file:
        json.dump(results, json_file)

    print("FINISHED")


if __name__ == "__main__":
    # BASE = "az_outputs_23aug/2023-08-20"

    # TODO: rerun for drugs and loan
    TABULAR_COMPLETED_JOBS = {
        "gretel": {
            "actgan": ["adult", "census", "child", "covtype", "credit", "insurance",
                       "intrusion", "drugs", "loan", "pums"]  # health_insurance
        },
        "sdv": {
            "gaussian_copula": ["adult", "census", "child", "covtype", "credit", "insurance",
                                "intrusion", "drugs", "loan", "pums", "health_insurance"],
            "ctgan": ["census", "child", "covtype", "credit",  # "insurance",
                      "intrusion", "drugs", "loan", "pums", "health_insurance"],
            # "adult",
            "tvae": ["adult", "census", "child", "covtype", "credit", "insurance",
                     "intrusion", "drugs", "loan", "pums", "health_insurance"]
        },
        "synthcity": {
            "ddpm": ["adult", "child", "covtype", "insurance",
                     "drugs", "loan", "health_insurance"],  # credit, intrusion, pums, census
            "arf": ["adult", "child", "covtype", "credit", "insurance",
                    "intrusion", "drugs", "loan", "health_insurance"],  # census@, pums
            "nflow": ["adult", "child", "covtype", "credit", "insurance",
                      "intrusion", "drugs", "loan", "pums", "health_insurance"], #censusX
            "goggle": ["adult", "child", "insurance",
                       "drugs", "loan", "health_insurance"],  # intrusion, covtype, census, credit, pums
            "rtvae": ["adult", "child", "covtype", "credit", "insurance",
                      "drugs", "loan", "health_insurance"],  # census, pums, intrusion
            "tvae": ["adult", "child", "credit", "insurance",
                     "intrusion", "drugs", "loan", "health_insurance"],  # census, pums, covtype
            "ctgan": ["adult", "child", "covtype", "credit", "insurance",
                      "intrusion", "drugs", "loan", "health_insurance"]  # census, pums
        },
        "llm": {
            "great": ["adult", "health_insurance", "loan"]
        },
        "betterdata": {
            # "fake_companies"], #,
            "gan": ["adult", "census", "loan", "health_insurance"],
            "gan_dp": ["adult", "loan"]
        },
        "hpo_synthcity": {
            "arf": ["adult", "loan"],
            "ctgan": ["adult", "loan"],
            "ddpm": ["adult", "loan"],
            "rtvae": ["adult", "loan"],
            "tvae": ["adult", "loan"]
        }

    }

    # TODO: Add sequential metrics
    # can plot sequential patterns
    SEQUENTIAL_COMPLETED_JOBS = {
        "gretel": {
            "dgan": ["taxi", "nasdaq", "pums"]
        },
        "sdv": {
            "par": ["taxi", "nasdaq"]
        }
    }

    lib = "gretel"  # "sdv"
    modality = "tabular"  # sequential



    parser = argparse.ArgumentParser()
    # parser.add_argument("--data_set", "--ds", type=str, default="s3",
    #                     help="enter set of data. \
    #                     Possible values - {s1, s2, s3}")
    parser.add_argument("--synthesizer", "--s", type=str, default="ctgan",
                        help="enter synthesizer name \
                            Possible values - {ctgan, tvae, gaussian_copula}")
    parser.add_argument("--output_folder", "--o",
                        type=str, default="metrics_out")

    args = parser.parse_args()

    # exp_data_set_name: str = args.data_set
    exp_synthesizer: str = args.synthesizer
    output_folder: str = args.output_folder

    # exp_synthesizer = "ctgan"  # gaussian_copula, tvae
    # TODO
    BASE_OUTPUT_PATH = f"{output_folder}/{modality}/{exp_synthesizer}"


    if modality in ["tabular", "hpo"]:
        BASE = f"final_outs/{lib}_tabular"
        exp_data_set = TABULAR_COMPLETED_JOBS[lib][exp_synthesizer]
    else:
        BASE = f"final_outs/sequential"
        exp_data_set = SEQUENTIAL_COMPLETED_JOBS[lib][exp_synthesizer]

    # if not os.path.exists(BASE_OUTPUT_PATH):
    #     os.makedirs(BASE_OUTPUT_PATH)

    # temp naming
    # if exp_data_set_name == "s1":
    #     exp_data_set = ["adult", "drugs", "intrusion"]
    # elif exp_data_set_name == "s2":
    #     exp_data_set = ["loan", "covtype", "child"]
    # elif exp_data_set_name == "llm":
    #     exp_data_set = ["adult", "health_insurance", "loan"]
    # else:
    #     exp_data_set = ["health_insurance", "census", "credit"]

    

    if not os.path.exists(BASE_OUTPUT_PATH):
        os.makedirs(BASE_OUTPUT_PATH)

    # exp_data_set_name = "xxx"
    logging.basicConfig(filename=f"{BASE_OUTPUT_PATH}/{exp_synthesizer}.log",
                        format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s ",
                        datefmt="%Y-%m-%d:%H:%M:%S",
                        filemode="w")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)

    exp_data_set = ["health_insurance"]

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
                        lib=lib)
        except Exception as e:
            print(e)
            LOGGER.error(e)
        LOGGER.info("*"*30)
        LOGGER.info(
            f"SUCCESS: Metrics generated for {dataset_name} {exp_synthesizer} ")
        LOGGER.info("*"*30)
