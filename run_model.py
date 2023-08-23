"""Setup configurations and run the synthesizer"""

import argparse
import logging
import os
from datetime import datetime

import pandas as pd

from commons.static_vals import (DEFAULT_EPOCH_VALUES,
                                 ML_CLASSIFICATION_TASK_DATASETS,
                                 ML_TASKS_TARGET_CLASS)
from commons.utils import (detect_metadata_with_sdv, get_dataset_with_sdv,
                           shuffle_and_split_dataframe,
                           stratified_split_dataframe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", "--l", type=str, default="sdv",
                        help="enter library. \
                        Possible values - {sdv, gretel}")
    parser.add_argument("--modality", "--m", type=str, default="tabular",
                        help="enter dataset modality. \
                        Possible values - {tabular, sequential, text}")
    parser.add_argument("--synthesizer", "--s", type=str, default="ctgan",
                        help="enter synthesizer name \
                            Possible values - {ctgan, tvae, gaussian_copula, par, dgan, actgan}")
    parser.add_argument("--data", type=str, default="adult",
                        help="enter dataset name. \
                        Possible values - {adult, census, child, covtype, \
                        credit, insurance, intrusion, health_insurance, drugs, loan, \
                        nasdaq, taxi, asu}")

    parser.add_argument("--use_gpu", "--gpu", type=bool, default=False,
                        help="whether to use GPU device(s)")
    parser.add_argument("--get_quality_report", "--qr", type=bool, default=False,
                        help="whether to generate SDV quality report")
    parser.add_argument("--get_diagnostic_report", "--dr", type=bool, default=False,
                        help="whether to generate SDV diagnostic report")
    # default epoch is set as 0 as statistical models do not need epochs
    parser.add_argument("--num_epochs", "--e", type=int, default=0)
    parser.add_argument("--data_folder", "--d", type=str, default="data")
    parser.add_argument("--output_folder", "--o", type=str, default="outputs")

    # -----------------------------------------------------------
    # parsing inputs
    # -----------------------------------------------------------
    args = parser.parse_args()

    exp_library: str = args.library
    exp_data_modality: str = args.modality
    exp_synthesizer: str = args.synthesizer
    exp_dataset_name: str = args.data
    
    print("args.use_gpu:  ------> ", args.use_gpu)
    use_gpu: bool = args.use_gpu
    num_epochs: int = args.num_epochs
    data_folder: str = args.data_folder
    output_folder: str = args.output_folder

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
            f"sample_datasets/personal_loan.csv")
        metadata = detect_metadata_with_sdv(real_dataset)
    else:
        real_dataset, metadata = get_dataset_with_sdv(
            "single_table", exp_dataset_name)

    train_dataset = real_dataset

    # --------------
    # Split and get training dataset
    # -------------
    if exp_data_modality == "tabular":
        if exp_dataset_name in ML_CLASSIFICATION_TASK_DATASETS:
            (X_train, y_train) = stratified_split_dataframe(real_dataset,
                                                            ML_TASKS_TARGET_CLASS[exp_dataset_name],
                                                            True)
            # Merge X_train and y_train columns horizontally
            train_dataset = pd.concat([X_train, y_train], axis=1)
        else:
            train_dataset = shuffle_and_split_dataframe(real_dataset, True)

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
