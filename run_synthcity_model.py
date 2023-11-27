# "Run Synthcity models -- CTGAN, TVAE, RTVAE, DDPM, GOGGLE, ARF, NFLOW"
from synthcity.utils.serialization import save_to_file
import json
import logging
# import pickle
# import sys
import os
import time
import tracemalloc
import warnings
# from io import StringIO
import torch
from commons.static_vals import N_BYTES_IN_MB, MLTasks, ML_TASKS_TARGET_CLASS

from synthcity.plugins import Plugins

print("Synthcity IMPORTED!")
# from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

# from synthcity.plugins.core.constraints import Constraints
# from synthcity.plugins.core.dataloader import GenericDataLoader

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# TODO: Improve this code :/


def run_model(**kwargs):
    data_modality = kwargs["exp_data_modality"]
    synthesizer_name = kwargs["exp_synthesizer"]
    dataset_name = kwargs["exp_data"]
    use_gpu = kwargs["use_gpu"]
    num_epochs = kwargs["num_epochs"]
    output_path = kwargs["output_path"]
    train_dataset = kwargs["train_dataset"]
    # metadata = kwargs["metadata"]
    num_samples = kwargs["num_samples"]
    ml_task = kwargs["ml_task"]
    opt_params = kwargs["opt_params"]
    # generate_sdv_quality_report = kwargs["get_quality_report"]
    # generate_sdv_diagnostic_report = kwargs["get_diagnostic_report"]

#     if kwargs["sequential_details"]:
#         num_sequences = kwargs["sequential_details"]["num_sequences"]
#         max_sequence_length = kwargs["sequential_details"]["max_sequence_length"]
#         seq_fixed_attributes = kwargs["sequential_details"]["fixed_attributes"]
#     synthesizer_class = SYNTHESIZERS_MAPPING[synthesizer_name]
#     # num_samples = len(train_dataset)

    device = "cuda" if use_gpu else "cpu"

    tracemalloc.start()

    is_classification = False
    target_class = None
    ddpm_cond = None

    # plugin_cls = type(Plugins().get(PLUGIN))
    # plugin = plugin_cls(**params).fit(train_loader)

    print("opt_params : ", opt_params)

    # if synthesizer_name == "goggle":
    #     # ---------------------
    #     # Generative MOdellinG with Graph LEarning (GOGGLE)
    #     # ---------------------
    #     # - Link to the paper: https://openreview.net/pdf?id=fPVRcJqspu
    #     synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
    #                                 device=torch.device(device))
    if synthesizer_name == "arf":
        # ---------------------
        # Adversarial Random Forests for Density Estimation and Generative Modeling
        # ---------------------
        # - Link to the paper: https://arxiv.org/pdf/2205.09435.pdf
        # synthesizer = Plugins().get(synthesizer_name, num_trees=50,
        #                             device=torch.device(device))
        if opt_params:
            print(f"setting optimised {synthesizer_name} parameters")
            synthesizer = Plugins().get(synthesizer_name, device=torch.device(device), **opt_params)
        else:
            print("setting default {synthesizer_name} parameters")
            synthesizer = Plugins().get(synthesizer_name, num_trees=50,
                                        device=torch.device(device))
    # elif synthesizer_name == "nflow":
    #     # ---------------------
    #     # Normalising Flow
    #     # ---------------------
    #     # - Link to the paper: https://arxiv.org/pdf/1906.04032.pdf
    #     synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
    #                                 device=torch.device(device))
    elif synthesizer_name == "ddpm":
        # ---------------------
        # Tabular denoising diffusion probabilistic models
        # ---------------------
        # - Link to the paper: https://arxiv.org/pdf/2209.15421.pdf
        if ml_task == MLTasks.CLASSIFICATION.value:
            is_classification = True
            # rename target column to 'target'
            target_class = ML_TASKS_TARGET_CLASS[dataset_name]
            train_dataset = train_dataset.rename(
                columns={target_class: "target"})
            print(f"target_class {target_class}, ml_task: {ml_task}")
        elif ml_task == MLTasks.REGRESSION.value:
            ddpm_cond = train_dataset[ML_TASKS_TARGET_CLASS[dataset_name]]

        # synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
        #                             is_classification=is_classification,
        #                             device=torch.device(device))
        if opt_params:
            print(f"setting optimised {synthesizer_name} parameters")
            synthesizer = Plugins().get(synthesizer_name, is_classification=is_classification,
                                        device=torch.device(device), **opt_params)
        else:
            print(f"setting default {synthesizer_name} parameters")
            synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
                                        is_classification=is_classification,
                                        device=torch.device(device))
    # elif synthesizer_name == "ctgan":
    #     synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
    #                                 device=torch.device(device))
    # elif synthesizer_name == "tvae":
    elif synthesizer_name in ["tvae", "ctgan", "rtvae", "nflow", "goggle"]:
        if opt_params:
            print(f"setting optimised {synthesizer_name} parameters")
            synthesizer = Plugins().get(synthesizer_name, device=torch.device(device), **opt_params)
        else:
            print("setting default {synthesizer_name} parameters")
            synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
                                        device=torch.device(device))
    elif synthesizer_name == "timegan":
        ...
    #     from synthcity.utils.datasets.time_series.google_stocks import GoogleStocksDataloader
    #     from synthcity.utils.datasets.time_series.pbc import PBCDataloader
    #     from synthcity.utils.datasets.time_series.sine import SineDataloader
    #     from synthcity.plugins.core.dataloader import TimeSeriesDataLoader
    #     from commons.seq_dataloader import sequential_data_loader

    #     # static, temporal, horizons, outcome = GoogleStocksDataloader().load()
    #     # static, temporal, horizons, outcome = PBCDataloader().load()
    #     # print(type(static), type(temporal), type(horizons), type(outcome))
    #     import pandas as pd
    #     # df = pd.read_csv("g.csv")
    #     train_dataset =  pd.read_csv("/Users/anshusingh/DPPCC/whitespace/benchmarking-synthetic-data-generators/data/sequential/nasdaq100_2019_mini.csv")
    #     static, temporal, horizons, outcome = sequential_data_loader(train_dataset, "Symbol", "Industry", "Date", ["Sector", "Industry"])
    #     # static_data, time_varying_data, observation_times, outcome_data

    #    # <class 'pandas.core.frame.DataFrame'> <class 'list'> <class 'list'> <class 'pandas.core.frame.DataFrame'>
    #     print("*"*10)
    #     print(len(static))
    #     print("*"*10)
    #     print(len(outcome), outcome)
    #     # breakpoint()
    #     print("*temporal"*10)
    #     print(len(temporal), temporal[0].shape, temporal[1].shape)
    #     print("*horizons"*10)
    #     print(len(horizons), len(horizons[0]), len(horizons[1]))
    #     print("*"*10)
    #     # breakpoint()
    #     loader = TimeSeriesDataLoader(
    #                  temporal_data=temporal, # list[DataFrame]
    #                  observation_times=horizons, #list
    #                 #  static_data=static, # dataframe
    #                 #  outcome=None, #dataframe
    #     )
    #     synthesizer = Plugins().get(synthesizer_name, n_iter=5)
    #     synthesizer.fit(loader)

    # elif synthesizer_name == "rtvae":
    #     synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs,
    #                                 device=torch.device(device))

#     LOGGER.info(synthesizer.get_parameters())

#     #  Store the print statements in a variable
#     captured_print_out = StringIO()
#     sys.stdout = captured_print_out

    begin_train_time = time.time()
    # ---------------------
    # Train
    # -------------------
    if ddpm_cond is not None:
        synthesizer.fit(train_dataset, cond=ddpm_cond)
    else:
        synthesizer.fit(train_dataset)
    end_train_time = time.time()

    # ---------------------
    # Sample
    # ---------------------
    begin_sampling_time = time.time()
    if ddpm_cond is not None:
        num_samples = len(train_dataset)
        synthetic_dataset = synthesizer.generate(
            count=num_samples, cond=ddpm_cond).dataframe()
    else:
        synthetic_dataset = synthesizer.generate(count=num_samples).dataframe()
    end_sampling_time = time.time()

    synthetic_dataset.to_csv(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv", index=False)

    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()

    # ---------------------
    # Prepare Outputs
    # ---------------------

    # sys.stdout = sys.__stdout__
    # captured_print_out = captured_print_out.getvalue()
    # print(captured_print_out)

    # ---------------------
    # Dump output to files
    # ---------------------
    # save print statements
    # with open(f"{output_path}{dataset_name}_{synthesizer_name}_out.txt", "w") as log_file:
    #     json.dump(captured_print_out, log_file)

    # Get the memory usage of the real and synthetic dataFrame in MB
    train_dataset_size_deep = train_dataset.memory_usage(
        deep=True).sum() / N_BYTES_IN_MB
    synthetic_dataset_size_deep = synthetic_dataset.memory_usage(
        deep=True).sum() / N_BYTES_IN_MB

    train_dataset_size = train_dataset.memory_usage(
        deep=False).sum() / N_BYTES_IN_MB
    synthetic_dataset_size = synthetic_dataset.memory_usage(
        deep=False).sum() / N_BYTES_IN_MB

    # Save model as Pytorch checkpoint
    save_to_file(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pkl", synthesizer)
    # synthesizer.save(
    #     f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pth")

    # Compute the size of the file in megabytes (MB)
    synthesizer_size = os.path.getsize(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pkl") / N_BYTES_IN_MB

    execution_scores = {
        # "Date": [today_date],
        "lib": f"synthcity==0.2.9",
        "modality": data_modality,
        "synthesizer": synthesizer_name,
        "optimised": True if opt_params else False,

        "dataset": dataset_name,
        "num_rows": train_dataset.shape[0],
        "num_cols": train_dataset.shape[1],
        "num_sampled_rows": num_samples,

        "device": device,
        "num_epochs": num_epochs if num_epochs else 0,
        # "num_cat": len(dataset_name),
        # "num_numeric": len(dataset_name),

        "train_time_sec": end_train_time - begin_train_time,
        "sample_time_sec": end_sampling_time - begin_sampling_time,

        "peak_memory_mb": peak_memory,
        "synthesizer_size": synthesizer_size,

        "synthetic_dataset_size_mb_deep": synthetic_dataset_size_deep,
        "train_dataset_size_mb_deep": train_dataset_size_deep,

        "synthetic_dataset_size_mb": synthetic_dataset_size,
        "train_dataset_size_mb": train_dataset_size
    }

    # ---------------------
    # Dump output to files
    # ---------------------
    # save print statements
    # with open(f"{output_path}{dataset_name}_{synthesizer_name}_out.txt", "w") as log_file:
    #     json.dump(captured_print_out, log_file)

    # save synthetic data

    if is_classification and synthesizer_name == "ddpm":
        # revert the name of the target column
        synthetic_dataset = synthetic_dataset.rename(
            columns={"target": target_class})

    synthetic_dataset.to_csv(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv", index=False)

    # save execution data
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_execution_scores.json", "w") as json_file:
        json.dump(execution_scores, json_file)

    print("-"*100)
    print(f"{synthesizer_name.upper()} trained  on {dataset_name.upper()} dataset| {num_samples} sampled | Files saved!")
    print("-"*100)
