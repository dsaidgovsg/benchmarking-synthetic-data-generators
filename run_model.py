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
                           stratified_split_dataframe)

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
                        nasdaq, taxi, asu}")
    parser.add_argument("--imputer", "--i", type=str, default="hyperimpute",
                        help="enter hyperimputer plugin name \
                            Possible values - {'simple', 'mice', 'ice', \
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
        output_path: str = f"{output_folder}/{today_date}/{exp_library}/{exp_data_modality}/{exp_synthesizer}_impute/{exp_dataset_name}/"
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

    if exp_data_modality == "sequential":

        if exp_dataset_name == "nasdaq":
            real_dataset, metadata = get_dataset_with_sdv(
                "sequential", "nasdaq100_2019")

            if exp_synthesizer == "dgan":
                # filter groups that have size 252 and drop the rest
                required_group_size = 252  # max-sequence length
                group_sizes = real_dataset.groupby('Symbol').size()
                valid_groups = group_sizes[group_sizes ==
                                           required_group_size].index
                real_dataset = real_dataset[real_dataset['Symbol'].isin(
                    valid_groups)]

            # Find minimum, maximum, and mean of the group sizes
            # grouped_sizes = df_filtered.groupby('Symbol').size()
            # min_size = grouped_sizes.min()
            # max_size = grouped_sizes.max()
            # mean_size = grouped_sizes.mean()
            # print("Minimum group size:", min_size)
            # print("Maximum group size:", max_size)
            # print("Mean group size:", mean_size, len(grouped_sizes), df_filtered.shape, real_dataset.shape)

            # Minimum group size: 177
            # Maximum group size: 252
            # Mean group size: 250.33009708737865
            # 252    100
            # 203      1
            # 204      1
            # 177      1

            sequential_details = {
                # drop 3 sequences for the dgan model, as it requires equal length of sequences
                "num_sequences": 100 if exp_synthesizer == "dgan" else 103,
                "max_sequence_length": 252,
                "fixed_attributes": ["Sector", "Industry"],  # context_columns,
                "varying_attributes": ["Open", "Close", "Volume", "MarketCap"],
                "time_attribute": "Date",
                "entity": "Symbol",
                "discrete_attributes": ["Sector", "Industry"]
            }
    # get accidential_drug_deaths.csv dataset from the local
    elif exp_dataset_name == "drugs":
        real_dataset = pd.read_csv(
            f"sample_datasets/accidential_drug_deaths.csv")
        metadata = detect_metadata_with_sdv(real_dataset)
    elif exp_dataset_name == "health_insurance":
        real_dataset = pd.read_csv(
            f"sample_datasets/health_insurance.csv")
        metadata = detect_metadata_with_sdv(real_dataset)
    elif exp_dataset_name == "loan":
        real_dataset = pd.read_csv(
            f"sample_datasets/loan.csv")
        metadata = detect_metadata_with_sdv(real_dataset)
    else:
        real_dataset, metadata = get_dataset_with_sdv(
            "single_table", exp_dataset_name)

    # --------------
    # Run hyperimpute if enabled
    # --------------
    if run_hyperimpute:
        print("Executing hyperimpute on train dataset")

        from run_hyperimpute import apply_imputation
        print(
            f"before imputation: total missing values in the real DataFrame: {real_dataset.isna().sum().sum()}")
        real_dataset = apply_imputation(dataframe=real_dataset,
                                        method=exp_imputer,
                                        dataset_name=exp_dataset_name,
                                        output_path=output_path)
        print(
            f"after imputation: total missing values in the real DataFrame: {real_dataset.isna().sum().sum()}")
        breakpoint()
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

    # --------------
    # Run models
    # --------------
    if exp_library == "sdv":
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

        run_model(
            exp_data_modality=exp_data_modality,
            exp_synthesizer=exp_synthesizer,
            exp_data=exp_dataset_name,
            use_gpu=use_gpu,
            num_epochs=num_epochs,
            train_dataset=train_dataset,
            metadata=metadata,
            output_path=output_path,
            sequential_details=sequential_details)

    elif exp_library == "synthcity":

        opt_params = {}
        if run_optimizer:
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
