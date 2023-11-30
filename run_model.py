"""Setup configurations and run the synthesizer"""

import argparse
import logging
import os
from datetime import datetime

import pandas as pd

from commons.static_vals import (DEFAULT_EPOCH_VALUES,
                                 MLTasks,
                                 ML_REGRESSION_TASK_DATASETS,
                                 ML_CLASSIFICATION_TASK_DATASETS,
                                 ML_TASKS_TARGET_CLASS)
from commons.utils import (detect_metadata_with_sdv, get_dataset_with_sdv,
                           shuffle_and_split_dataframe,
                           stratified_split_dataframe,
                           split_sequential_data)
from commons.sequential import (process_groups, get_groups_stats)

# Note: ACTGAN model from gretel-synthetics requires sdv<0.18
# TODO: rename TABULAR to {STANDARD, GENERIC}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", "--l", type=str, default="sdv",
                        help="enter library. \
                        Possible values - {sdv, gretel, synthcity}")
    parser.add_argument("--modality", "--m", type=str, default="tabular",
                        help="enter dataset modality. \
                        Possible values - {tabular, sequential, text}")
    parser.add_argument("--synthesizer", "--s", type=str, default="ctgan",
                        help="enter synthesizer name \
                            Possible values - {ctgan, tvae, gaussian_copula, par, dgan, actgan, goggle}")
    parser.add_argument("--data", type=str, default="adult",
                        help="enter dataset name. \
                        Possible values - {adult, census, child, covtype, \
                        credit, insurance, intrusion, health_insurance, drugs, loan, \
                        nasdaq, taxi, pums}")  # IL_OH_10Y_PUMS
    parser.add_argument("--imputer", "--i", type=str, default="hyperimpute",
                        help="enter hyperimputer plugin name \
                            Possible values - {'simple', 'mice',  \
                            'missforest', 'hyperimpute'")
    # Possible values - {'median', 'sklearn_ice', 'mice', 'nop', \
    # 'missforest', 'EM', 'ice', 'most_frequent', 'mean', \
    # 'miracle', 'miwae', 'hyperimpute', 'gain', 'sklearn_missforest', \
    # 'softimpute', 'sinkhorn'")
    parser.add_argument("--optimizer_trials", "--trials", type=int, default=25)

    # default epoch is set as 0 as statistical models do not need epochs
    parser.add_argument("--num_epochs", "--e", type=int, default=0)
    parser.add_argument("--data_folder", "--d", type=str, default="data")
    parser.add_argument("--output_folder", "--o", type=str, default="outputs")

    # boolean args
    parser.add_argument("--train_test_data", "--tt", default=False,
                        help="whether to return train and test data", action='store_true')
    parser.add_argument("--get_quality_report", "--qr", default=False,
                        help="whether to generate SDV quality report", action='store_true')
    parser.add_argument("--get_diagnostic_report", "--dr", default=False,
                        help="whether to generate SDV diagnostic report", action='store_true')
    parser.add_argument("--run_optimizer", "--ro", default=False,
                        help="whether to run hyperparameter optimizer", action='store_true')
    parser.add_argument("--run_hyperimpute", "--ri", default=False,
                        help="whether to run hyperimpute", action='store_true')
    parser.add_argument("--run_model_training", "--rt", default=False,
                        help="whether to train a model", action='store_true')
    parser.add_argument("--use_gpu", "--cuda", default=False,
                        help="whether to use GPU device(s)", action='store_true')

    args = parser.parse_args()

    print("Arguments: ", vars(args))
    # print("train_test_data--->", args.train_test_data)
    # print("get_quality_report--->", args.get_quality_report)
    # print("get_diagnostic_report--->", args.get_diagnostic_report)
    # print("run_optimizer--->", args.run_optimizer)
    # -----------------------------------------------------------
    # parsing inputs
    # -----------------------------------------------------------

    exp_library: str = args.library
    exp_data_modality: str = args.modality
    exp_synthesizer: str = args.synthesizer
    exp_dataset_name: str = args.data
    exp_imputer: str = args.imputer

    use_gpu: bool = args.use_gpu
    num_epochs: int = args.num_epochs
    optimizer_trials: int = args.optimizer_trials
    data_folder: str = args.data_folder
    output_folder: str = args.output_folder
    train_test_data: bool = args.train_test_data
    run_optimizer: bool = args.run_optimizer
    run_hyperimpute: bool = args.run_hyperimpute
    run_model_training: bool = args.run_model_training

    get_quality_report: bool = args.get_quality_report
    get_diagnostic_report: bool = args.get_diagnostic_report

    # TODO: Add assertions to validate the inputs
    # assert exp_data_modality in VALID_DATA_MODALITIES
    # assert exp_synthesizer in VALID_MODALITY_SYNTHESIZER
    # assert exp_data in VALID_MODALITY_DATA

    today_date: str = datetime.now().strftime("%Y-%m-%d")
    # -----------------------------------------------------------
    # Output folder for saving:
    # 1.Logs 2.Synthetic Data 3.Execution Metrics 4.Saved Models
    # -----------------------------------------------------------

    if args.run_hyperimpute:
        output_path: str = f"{output_folder}/{today_date}/{exp_library}/{exp_data_modality}/{exp_synthesizer}_{exp_imputer}/{exp_dataset_name}/"
    elif args.run_optimizer:
        output_path: str = f"{output_folder}/{today_date}/{exp_library}/{exp_data_modality}/{exp_synthesizer}_hpo/{exp_dataset_name}/"
    else:
        output_path: str = f"{output_folder}/{today_date}/{exp_library}/{exp_data_modality}/{exp_synthesizer}/{exp_dataset_name}/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(filename=f"{output_path}{exp_dataset_name}_{exp_synthesizer}.log",
                        format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s ",
                        datefmt="%Y-%m-%d:%H:%M:%S",
                        filemode="w")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)

    # --------------
    # Get dataset
    # --------------
    # sequential datasets require extra parameters
    sequential_details: dict = None
    print("-"*30)
    print("Loading dataset")
    print("-"*30)

    if exp_data_modality == "sequential":

        # DGAN requires equal length of the sequences; Apply operation:
        # 1. drop_and_truncate 2. padding_and_truncate

        if exp_dataset_name == "nasdaq":
            # Minimum group size: 177
            # Maximum group size: 252
            # Mean group size: 250.33009708737865
            # 252    100
            # 203      1
            # 204      1
            # 177      1
            try:
                real_dataset = pd.read_csv(
                    f"sample_datasets/seq/{exp_dataset_name}.csv")
                real_dataset.drop(columns=["Unnamed: 0"], inplace=True)
                metadata = detect_metadata_with_sdv(real_dataset)
            except Exception as e:
                real_dataset, metadata = get_dataset_with_sdv(
                    "sequential", "nasdaq100_2019")

            # max_sequence_length = 252  # max-sequence length
            req_sequence_length = 250
            # padding_and_truncate, drop_and_truncate
            groups_processing_op = "drop_and_truncate"
            # TODO: padding_and_truncate needs to be updated to retain static values consistency in a group

            entity_col = "Symbol"
            temporal_col = "Date"
            static_cols = ["Sector", "Industry"]
            dynamic_cols = ["Open", "Close", "Volume", "MarketCap"]
            discrete_cols = ["Sector", "Industry"]

        elif exp_dataset_name == "taxi":

            # padding_and_truncate, drop_and_truncate

            real_dataset = pd.read_csv(
                f"sample_datasets/seq/{exp_dataset_name}.csv")

            # real_dataset.fillna(0, inplace=True)

            metadata = detect_metadata_with_sdv(real_dataset)
            print(metadata)

            req_sequence_length = 90  # TODO update
            groups_processing_op = "drop_and_truncate"

            entity_col = "taxi_id"
            temporal_col = "trip_start_timestamp"

            static_cols = []
            dynamic_cols = [
                'trip_end_timestamp', 'trip_seconds', 'trip_miles',
                'pickup_census_tract', 'dropoff_census_tract', 'pickup_community_area',
                'dropoff_community_area', 'fare', 'tips', 'tolls', 'extras', 'trip_total',
                'payment_type', 'pickup_latitude', 'pickup_longitude', "company",
                'dropoff_latitude', 'dropoff_longitude'
            ]

            discrete_cols = [
                "pickup_census_tract",  # If represented as strings or categorical codes
                "dropoff_census_tract",  # Same as above
                "pickup_community_area",  # If these are categorical codes
                "dropoff_community_area",  # Same as above
                # Typically a string (e.g., 'Cash', 'Credit Card')
                "payment_type",
                "company",  # Company names or IDs as strings or categories
                # If latitude and longitude are categorical codes rather than numerical coordinates:
                "pickup_latitude",
                "pickup_longitude",
                "dropoff_latitude",
                "dropoff_longitude"
            ]

        elif exp_dataset_name == "pums":

            real_dataset = pd.read_csv(
                f"sample_datasets/seq/{exp_dataset_name}.csv")
            real_dataset.drop(columns=["Unnamed: 0"], inplace=True)
            # real_dataset.fillna(0, inplace=True)

            metadata = detect_metadata_with_sdv(real_dataset)
            print(metadata)

            req_sequence_length = 6  # TODO update
            groups_processing_op = "drop_and_truncate"

            # Entity Column
            # This variable identifies unique entities (individuals) in the dataset.
            entity_col = 'sim_individual_id'

            # Temporal Column
            # This variable represents the time dimension in the dataset.
            temporal_col = 'YEAR'

            # Static Variables
            # These variables do not change over time for a given individual.
            static_cols = ['SEX', 'RACE', 'HISPAN']  # sim_individual_id

            # Dynamic Variables
            # These variables can change over time for a given individual.
            dynamic_cols = ['YEAR', 'HHWT', 'GQ', 'PERWT', 'AGE', 'MARST', 'SPEAKENG', 'HCOVANY',
                            'HCOVPRIV', 'HINSEMP', 'HINSCAID', 'HINSCARE', 'EMPSTAT', 'EMPSTATD',
                            'LABFORCE', 'WRKLSTWK', 'ABSENT', 'LOOKING', 'AVAILBLE', 'WRKRECAL',
                            'WORKEDYR', 'INCTOT', 'INCWAGE', 'INCWELFR', 'INCINVST', 'INCEARN',
                            'POVERTY', 'DEPARTS', 'ARRIVES', 'CITIZEN', 'EDUC']
            
            discrete_cols = None

            # discrete_cols = [
            #     'PUMA',       # Public Use Microdata Area code (Categorical)
            #     'YEAR',       # Year of the survey (Categorical, though numerical, it represents distinct time periods)
            #     'GQ',         # Group Quarters status (Categorical)
            #     'SEX',        # Gender (Categorical/Binary)
            #     'MARST',      # Marital status (Categorical)
            #     'RACE',       # Race (Categorical)
            #     'HISPAN',     # Hispanic origin (Categorical)
            #     'CITIZEN',    # Citizenship status (Categorical)
            #     'SPEAKENG',   # English speaking ability (Categorical)
            #     'HCOVANY',    # Healthcare coverage status (Categorical/Binary)
            #     'HCOVPRIV',   # Private health insurance status (Categorical/Binary)
            #     'HINSEMP',    # Health insurance through employment (Categorical/Binary)
            #     'HINSCAID',   # Health insurance through Medicaid (Categorical/Binary)
            #     'HINSCARE',   # Health insurance through Medicare (Categorical/Binary)
            #     'EDUC',       # Education level (Categorical)
            #     'EMPSTAT',    # Employment status (Categorical)
            #     'EMPSTATD',   # Detailed employment status (Categorical)
            #     'LABFORCE',   # Labor force status (Categorical)
            #     'WRKLSTWK',   # Worked last week status (Categorical)
            #     'ABSENT',     # Absent from work status (Categorical)
            #     'LOOKING',    # Looking for work status (Categorical)
            #     'AVAILBLE',   # Availability for work status (Categorical)
            #     'WRKRECAL',   # Work recall status (Categorical)
            #     'WORKEDYR',   # Worked this year status (Categorical/Binary)
            #     'POVERTY'     # Poverty status indicator (Categorical)
            # ]


        # exp_synthesizer == "dgan"
        if groups_processing_op:
            get_groups_stats(real_dataset, entity_col)
            print("Number of sequences BEFORE processing: ",
                  real_dataset.groupby(entity_col).size().count())
            real_dataset = process_groups(real_dataset,
                                          req_sequence_length,
                                          entity_col,
                                          operation=groups_processing_op)
            get_groups_stats(real_dataset, entity_col)

            # valid_groups = group_sizes[group_sizes ==
            #                            max_sequence_length].index
            # real_dataset = real_dataset[real_dataset['Symbol'].isin(
            #     valid_groups)]

            # Assuming df is your DataFrame
            # df = pd.read_csv('your_data.csv') # Example to load data from a CSV file

            # Group by 'taxi_id' and filter out groups with more than one unique 'company'
            # grouped = real_dataset.groupby('taxi_id')['company'].nunique()
            # consistent_taxi_ids = grouped[grouped == 1].index
            # real_dataset = real_dataset[real_dataset['taxi_id'].isin(consistent_taxi_ids)]

            # real_dataset.fillna(0, inplace=True)

            # for col in real_dataset.columns:
            #     real_dataset[col] = pd.to_numeric(
            #         real_dataset[col], errors='coerce')

            num_sequences = real_dataset.groupby(entity_col).size().count()
            print("Number of sequences AFTER processing: ", num_sequences)

        sequential_details = {
            # drop 3 sequences for the dgan model, as it requires equal length of sequences
            "num_sequences": num_sequences,
            "max_sequence_length": req_sequence_length,
            "static_attributes": static_cols,
            "dynamic_attributes": dynamic_cols,
            "time_attribute": temporal_col,
            "entity": entity_col,
            "discrete_attributes": discrete_cols
        }

    else:
        metadata = None
        try:
            real_dataset = pd.read_csv(
                f"sample_datasets/{exp_dataset_name}.csv")
        except:
            real_dataset, metadata = get_dataset_with_sdv(
                "single_table", exp_dataset_name)

        # Drop the column: removing PII
        if exp_dataset_name in ["loan", "drugs"]:
            # Check if the column exists in DataFrame
            if "ID" in real_dataset.columns:
                try:
                    real_dataset.drop(columns=["ID"], inplace=True)
                except Exception as e:
                    print(e)

        # if exp_dataset_name == "pums":
        # Check if the column exists in DataFrame
        if "Unnamed: 0" in real_dataset.columns:
            try:
                real_dataset.drop(columns=["Unnamed: 0"], inplace=True)
            except Exception as e:
                print(e)

        # ACTGAN requires SDV < 0.18 that does not support metadata detection API
        if exp_synthesizer != "actgan" and not metadata:
            metadata = detect_metadata_with_sdv(real_dataset)

    print(real_dataset.columns)
    # elif exp_dataset_name == "drugs":
    #     real_dataset = pd.read_csv(
    #         f"sample_datasets/drugs.csv")
    #     metadata = detect_metadata_with_sdv(
    #         real_dataset) if not exp_synthesizer == "actgan" else None
    # elif exp_dataset_name == "health_insurance":
    #     real_dataset = pd.read_csv(
    #         f"sample_datasets/health_insurance.csv")
    #     metadata = detect_metadata_with_sdv(real_dataset)
    # elif exp_dataset_name == "loan":
    #     real_dataset = pd.read_csv(
    #         f"sample_datasets/loan.csv")
    #     metadata = detect_metadata_with_sdv(
    #         real_dataset) if not exp_synthesizer == "actgan" else None
    # else:
    #     if exp_synthesizer == "actgan":
    #         real_dataset, metadata = get_dataset_with_sdv(
    #             "single_table", exp_dataset_name) if not exp_synthesizer == "actgan" else None

    # --------------
    # Run hyperimpute if enabled
    # --------------
    if run_hyperimpute:
        print("-"*30)
        print("Running imputation")
        print("-"*30)
        print("Executing hyperimpute on train dataset")

        from run_hyperimpute import apply_imputation
        print(
            f"before imputation: percent missing values in the real DataFrame: \
                {real_dataset.isna().sum().sum()/(real_dataset.shape[0]*real_dataset.shape[1])*100}")
        real_dataset = apply_imputation(dataframe=real_dataset,
                                        method=exp_imputer,
                                        dataset_name=exp_dataset_name,
                                        output_path=output_path)
        print(
            f"after imputation: total missing values in the real DataFrame: \
                {real_dataset.isna().sum().sum()}")
    # else:
    #     train_dataset = real_dataset

    # breakpoint()
    # todo pass the real-dataset

    # --------------
    # Split and get training dataset
    # -------------
    if exp_data_modality == "tabular":
        if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
            if not train_test_data:
                (X_train, y_train) = stratified_split_dataframe(real_dataset,
                                                                ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                                                True)
                # Merge X_train and y_train columns horizontally
                train_dataset = pd.concat([X_train, y_train], axis=1)
            else:
                (X_train, y_train), (X_test, y_test) = stratified_split_dataframe(real_dataset,
                                                                                  ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                                                                  False)
                # Merge columns horizontally
                train_dataset = pd.concat([X_train, y_train], axis=1)
                test_dataset = pd.concat([X_test, y_test], axis=1)
        else:
            if not train_test_data:
                train_dataset = shuffle_and_split_dataframe(real_dataset, True)
            else:
                (train_dataset, test_dataset) = shuffle_and_split_dataframe(
                    real_dataset, False)
    elif exp_data_modality == "sequential":
        if not train_test_data:
            train_dataset = split_sequential_data(
                real_dataset, entity_col, True)
        else:
            (train_dataset, test_dataset) = split_sequential_data(
                real_dataset, entity_col, False)
    else:
        train_dataset = real_dataset

    # --------------
    # Run models
    # --------------
    if run_model_training:
        print("-"*30)
        print("Starting model training")
        print("-"*30)
        if exp_library == "sdv":
            print(
                f"INSIDE SDV: total missing values in the real DataFrame: {real_dataset.isna().sum().sum()}")
            from run_sdv_model import run_model

            if not num_epochs:
                num_epochs = DEFAULT_EPOCH_VALUES["sdv"][exp_synthesizer]

            print("Selected Synthesizer Library: SDV")
            LOGGER.info(
                (f"Modality: {exp_data_modality} | Synthesizer: {exp_synthesizer} | Dataset: {exp_dataset_name} | Epochs: {num_epochs}"))

            # print(real_dataset.shape, train_dataset.shape)
            LOGGER.info(
                f"Real dataset: {real_dataset.shape}, Train dataset: {train_dataset.shape}")

            num_samples = len(real_dataset)
            run_model(
                exp_data_modality=exp_data_modality,
                exp_synthesizer=exp_synthesizer,
                exp_data=exp_dataset_name,
                use_gpu=use_gpu,
                num_epochs=num_epochs,
                train_dataset=train_dataset,
                metadata=metadata,
                num_samples=num_samples,
                output_path=output_path,
                sequential_details=sequential_details,
                get_quality_report=get_quality_report,
                get_diagnostic_report=get_diagnostic_report)

        elif exp_library == "gretel":
            from run_gretel_model import run_model

            if not num_epochs:
                num_epochs = DEFAULT_EPOCH_VALUES["gretel"][exp_synthesizer]

            print("Selected Synthesizer Library: GRETEL")
            LOGGER.info(
                (f"Modality: {exp_data_modality} | Synthesizer: {exp_synthesizer} | Dataset: {exp_dataset_name} | Epochs: {num_epochs}"))
            LOGGER.info(
                f"Real dataset: {real_dataset.shape}, Train dataset: {train_dataset.shape}")

            run_model(
                exp_data_modality=exp_data_modality,
                exp_synthesizer=exp_synthesizer,
                exp_data=exp_dataset_name,
                use_gpu=use_gpu,
                num_epochs=num_epochs,
                train_dataset=train_dataset,
                # metadata=metadata,
                output_path=output_path,
                sequential_details=sequential_details)

        elif exp_library == "synthcity":

            print("$$"*10)
            print(
                f"after imputation: total missing values in the real DataFrame: \
                {real_dataset.isna().sum().sum()}")
            print("%%"*10)

            opt_params = {}
            if run_optimizer:
                print("-"*30)
                print("Fetching optimal hyperparameters.")
                print("-"*30)
                from run_synthcity_hpo import run_synthcity_optimizer
                print("Running Hyperparamter Optimisation")
                opt_params = run_synthcity_optimizer(exp_synthesizer,
                                                     exp_dataset_name,
                                                     train_dataset,
                                                     test_dataset,
                                                     output_path,
                                                     optimizer_trials)

                if not opt_params:
                    raise ("Hyperparamter optimisation failed!")
                else:
                    print("Here are the params: ", opt_params)

            if not num_epochs:
                num_epochs = DEFAULT_EPOCH_VALUES["synthcity"][exp_synthesizer]

            print("Selected Synthesizer Library: SYNTHCITY")
            LOGGER.info(
                (f"Modality: {exp_data_modality} | Synthesizer: {exp_synthesizer} | Dataset: {exp_dataset_name} | Epochs: {num_epochs}"))

            from run_synthcity_model import run_model

            num_samples = len(real_dataset)

            # flag for classification or regression (required by DDPM model)
            ml_task = None
            if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
                ml_task = MLTasks.CLASSIFICATION.value
            elif exp_dataset_name in ML_REGRESSION_TASK_DATASETS:
                ml_task = MLTasks.REGRESSION.value

            run_model(
                exp_data_modality=exp_data_modality,
                exp_synthesizer=exp_synthesizer,
                exp_data=exp_dataset_name,
                use_gpu=use_gpu,
                num_samples=num_samples,
                num_epochs=num_epochs,
                train_dataset=train_dataset,
                metadata=metadata,
                output_path=output_path,
                ml_task=ml_task,
                opt_params=opt_params)
