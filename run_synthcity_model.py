# "Run Synthcity models -- CTGAN, TVAE, RTVAE, DDPM, GOGGLE, ARF, NFLOW"
import json
import logging
# import pickle
# import sys
import os
import time
import tracemalloc
import warnings
from io import StringIO

from commons.static_vals import N_BYTES_IN_MB, DataModalities

from synthcity.plugins import Plugins
from synthcity.utils.serialization import save_to_file, load_from_file
# from synthcity.plugins.core.dataloader import DataLoader, GenericDataLoader

# from synthcity.plugins.core.constraints import Constraints
# from synthcity.plugins.core.dataloader import GenericDataLoader

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)


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
    # generate_sdv_quality_report = kwargs["get_quality_report"]
    # generate_sdv_diagnostic_report = kwargs["get_diagnostic_report"]

#     if kwargs["sequential_details"]:
#         num_sequences = kwargs["sequential_details"]["num_sequences"]
#         max_sequence_length = kwargs["sequential_details"]["max_sequence_length"]
#         seq_fixed_attributes = kwargs["sequential_details"]["fixed_attributes"]
#     synthesizer_class = SYNTHESIZERS_MAPPING[synthesizer_name]
#     # num_samples = len(train_dataset)

    tracemalloc.start()
    print("0"*10, synthesizer_name)

    # TODO: Add parameters
    if synthesizer_name == "goggle":
        # ---------------------
        # Generative MOdellinG with Graph LEarning (GOGGLE)
        # ---------------------
        # - Link to the paper: https://openreview.net/pdf?id=fPVRcJqspu
        synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs)
    elif synthesizer_name == "arf":
        # ---------------------
        # Adversarial Random Forests for Density Estimation and Generative Modeling
        # ---------------------
        # - Link to the paper: https://arxiv.org/pdf/2205.09435.pdf
        synthesizer = Plugins().get(synthesizer_name, num_trees=50)
    elif synthesizer_name == "nflow":
        # ---------------------
        # Normalising Flow
        # ---------------------
        # - Link to the paper: https://arxiv.org/pdf/1906.04032.pdf
        synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs)
    elif synthesizer_name == "ddpm":
        # ---------------------
        # Tabular denoising diffusion probabilistic models
        # ---------------------
        # - Link to the paper: https://arxiv.org/pdf/2209.15421.pdf
        synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs)
    elif synthesizer_name == "ctgan":
        synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs)
    elif synthesizer_name == "tvae":
        synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs)
    elif synthesizer_name == "rtvae":
        synthesizer = Plugins().get(synthesizer_name, n_iter=num_epochs)


#     LOGGER.info(synthesizer.get_parameters())

#     #  Store the print statements in a variable
#     captured_print_out = StringIO()
#     sys.stdout = captured_print_out

    begin_train_time = time.time()
    # ---------------------
    # Train
    # ---------------------
    synthesizer.fit(train_dataset)
    end_train_time = time.time()

    # ---------------------
    # Sample
    # ---------------------
    begin_sampling_time = time.time()
    synthetic_dataset = synthesizer.generate(count=num_samples).dataframe()
    end_sampling_time = time.time()

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

        "dataset": dataset_name,
        "num_rows": train_dataset.shape[0],
        "num_cols": train_dataset.shape[1],
        "num_sampled_rows": num_samples,

        "device": "GPU" if use_gpu else "CPU",
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
    synthetic_dataset.to_csv(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv")

    # save execution data
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_execution_scores.json", "w") as json_file:
        json.dump(execution_scores, json_file)

    print("-"*100)
    print(f"{synthesizer_name.upper()} trained  on {dataset_name.upper()} dataset| {num_samples} sampled | Files saved!")
    print("-"*100)
