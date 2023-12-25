"Run SDV models -- CTGAN, TVAE, Gaussin Copula, PAR"
import json
import logging
import pickle
import sys
import time
import tracemalloc
import warnings
from io import StringIO

import sdv

from commons.static_vals import N_BYTES_IN_MB, DataModalities
# from metrics.sdv_reports import (compute_sdv_quality_report,
#                                  compute_sdv_diagnostic_report)
# SDV synthesizers
# from synthesizers.sdv.sequential.par_synthesizer import PARSynthesizer

from sdv.sequential import PARSynthesizer
from sdv.single_table import CTGANSynthesizer, TVAESynthesizer, GaussianCopulaSynthesizer

# from synthesizers.sdv.tabular.copulas_synthesizer import \
#     GaussianCopulaSynthesizer
# from synthesizers.sdv.tabular.gen_synthesizer import (CTGANSynthesizer,
#                                                       TVAESynthesizer)

# from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

# from sdv.single_table import TVAESynthesizer
# from sdv.single_table import CTGANSynthesizer
# from sdv.single_table import GaussianCopulaSynthesizer
# from sdv.sequential import PARSynthesizer

# Link to Transform and Anonymize
# https://docs.sdv.dev/sdv/single-table-data/modeling/synthetic-data-workflow/transform-and-anonymize

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

SYNTHESIZERS_MAPPING = {
    # The CTGAN Synthesizer uses GAN-based DL methods
    "ctgan": CTGANSynthesizer,
    # The TVAE Synthesizer uses a variational autoencoder (VAE)-based DL methods
    "tvae": TVAESynthesizer,
    # The Gaussian Copula Synthesizer uses statistical methods
    "gaussian_copula": GaussianCopulaSynthesizer,
    # The PARSynthesizer uses a DL methods
    "par": PARSynthesizer
}


def run_model(**kwargs):
    data_modality = kwargs["exp_data_modality"]
    synthesizer_name = kwargs["exp_synthesizer"]
    dataset_name = kwargs["exp_data"]
    use_gpu = kwargs["use_gpu"]
    num_epochs = kwargs["num_epochs"]
    output_path = kwargs["output_path"]
    train_dataset = kwargs["train_dataset"]
    metadata = kwargs["metadata"]
    num_samples = kwargs["num_samples"]
    generate_sdv_quality_report = kwargs["get_quality_report"]
    generate_sdv_diagnostic_report = kwargs["get_diagnostic_report"]
    # num_samples=kwargs["num_samples"]

    if kwargs["sequential_details"]:

        entity_col = kwargs["sequential_details"]["entity"]
        temporal_col = kwargs["sequential_details"]["time_attribute"]

        num_sequences = kwargs["sequential_details"]["num_sequences"]
        max_sequence_length = kwargs["sequential_details"]["max_sequence_length"]
        seq_static_attributes = kwargs["sequential_details"]["static_attributes"]

    synthesizer_class = SYNTHESIZERS_MAPPING[synthesizer_name]
    # num_samples = len(train_dataset)

    # TODO: Might need to save metadata JSON file for some datasets
    # metadata = detect_metadata(real_data)
    LOGGER.info(metadata)

    tracemalloc.start()

    if synthesizer_name == "gaussian_copula":
        # ---------------------
        # Gaussain Copula
        # ---------------------
        # - Learns - (1) Marginal Distributions (2) Covariance
        # - Link to the paper: https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf
        # Params:
        # numerical_distributions = {
        #     <column name>: "norm",
        # },
        # One of: "norm" "beta", "truncnorm", "uniform", "gamma" or "gaussian_kde"
        # default_distribution <str> ="beta"
        synthesizer = synthesizer_class(metadata, 
                                        enforce_min_max_values=True,
                                        enforce_rounding=False,
                                        numerical_distributions={
                                            'amenities_fee': 'beta',
                                            'checkin_date': 'uniform'
                                        },
                                        default_distribution='norm')
    elif synthesizer_name == "par":  # sequential
        # Note: PAR model works for varying sequence lengths
        # ---------------------------------------------
        # PAR: Probabilistic Auto-Regressive model
        # ---------------------------------------------
        # - Learns how to create brand new sequences of multi-dimensional data,
        #   by conditioning on the unchanging, context values.
        # - Models non-numerical columns, including columns with missing values.
        # - Designed to work on multi-sequence data
        # - Link to the paper: https://arxiv.org/pdf/2207.14406.pdf
        # Params:
        # context_columns <list>: do not vary inside of a sequence.
        metadata.update_column(
            column_name=entity_col,
            sdtype='id'
        )

        metadata.set_sequence_key(column_name=entity_col)
        if dataset_name in "pums":
            metadata.update_column(
                column_name=temporal_col,
                sdtype='numerical'
            )
        metadata.set_sequence_index(column_name=temporal_col)

        print(">"*20)
        print(metadata.to_dict())
        print(">"*20)
        
        synthesizer = synthesizer_class(metadata,
                                        context_columns=seq_static_attributes,
                                        enforce_rounding=True,
                                        enforce_min_max_values=True,
                                        epochs=num_epochs,
                                        cuda=use_gpu,
                                        verbose=True)

    elif synthesizer_name == "ctgan":
        # --------------------------
        # CTGAN: Conditional Table GAN
        # --------------------------
        # - Link to the paper: https://arxiv.org/pdf/1907.00503.pdf
        # Params:
        # epochs: The optimal number of epochs depends on both the complexity of your dataset
        #         and the metrics you are using to quantify success.
        # links: https://github.com/sdv-dev/SDV/discussions/980
        #        https://datacebo.com/blog/interpreting-ctgan-progress/

        # metadata = metadata.to_dict()
        # metadata[""]
        # from sdv.metadata import SingleTableMetadata
        # metadata = SingleTableMetadata.load_from_dict(metadata)
        synthesizer = synthesizer_class(metadata,
                                        epochs=num_epochs,
                                        cuda=use_gpu,
                                        verbose=True)
    elif synthesizer_name == "tvae":
        # --------------------------
        # TVAE
        # --------------------------
        # - Link to the paper: https://arxiv.org/pdf/1907.00503.pdf
        synthesizer = synthesizer_class(metadata,
                                        epochs=num_epochs,
                                        cuda=use_gpu)

    LOGGER.info(synthesizer.get_parameters())

    #  Store the print statements in a variable
    captured_print_out = StringIO()
    sys.stdout = captured_print_out

    begin_train_time = time.time()
    # ---------------------
    # Train
    # ---------------------
    synthesizer.fit(train_dataset)
    end_train_time = time.time()

    if synthesizer_name == "gaussian_copula":
        LOGGER.info(synthesizer.get_learned_distributions())
    # ---------------------
    # Sample
    # ---------------------
    if synthesizer_name == "par":
        # Params:
        # num_sequences: An integer >0 describing the number of sequences to sample
        # sequence_length:
        #     An integer >0 describing the length of each sequence.
        #     If you provide None, the synthesizer will determine the lengths algorithmically
        #     , and the length may be different for each sequence. Defaults to None.
        begin_sampling_time = time.time()
        print("1"*10)
        synthetic_dataset = synthesizer.sample(num_sequences=num_sequences,
                                               sequence_length=max_sequence_length)
        print("2"*10)
    else:
        begin_sampling_time = time.time()
        synthetic_dataset = synthesizer.sample(num_rows=num_samples)
    end_sampling_time = time.time()

    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()

    # ---------------------
    # Prepare Outputs
    # ---------------------
    synthesizer_size = len(pickle.dumps(synthesizer)) / N_BYTES_IN_MB

    # Get the memory usage of the real and synthetic dataFrame in MB
    train_dataset_size_deep = train_dataset.memory_usage(
        deep=True).sum() / N_BYTES_IN_MB
    synthetic_dataset_size_deep = synthetic_dataset.memory_usage(
        deep=True).sum() / N_BYTES_IN_MB

    train_dataset_size = train_dataset.memory_usage(
        deep=False).sum() / N_BYTES_IN_MB
    synthetic_dataset_size = synthetic_dataset.memory_usage(
        deep=False).sum() / N_BYTES_IN_MB

    execution_scores = {
        # "Date": [today_date],
        "lib": f"SDV_{sdv.__version__}",
        "modality": data_modality,
        "synthesizer": synthesizer_name,

        "dataset": dataset_name,
        "num_rows": train_dataset.shape[0],
        "num_cols": train_dataset.shape[1],

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

    sys.stdout = sys.__stdout__
    captured_print_out = captured_print_out.getvalue()
    print(captured_print_out)

    # ---------------------
    # Dump output to files
    # ---------------------
    # save print statements
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_out.txt", "w") as log_file:
        json.dump(captured_print_out, log_file)

    # save model
    synthesizer.save(
        filepath=f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pkl"
    )
    # save synthetic data
    synthetic_dataset.to_csv(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv", index=False)

    # if generate_sdv_quality_report:
    #     print("Generating Quality Report", "#"*10)

    #     start_time = time.time()
    #     quality_report_obj = compute_sdv_quality_report(
    #         train_dataset,
    #         synthetic_dataset,
    #         metadata
    #     )
    #     execution_scores["quality_report_time_sec"] = time.time() - start_time

    #     print("Quality: ", quality_report_obj["score"])
    #     LOGGER.info("Quality: ", quality_report_obj["score"])

    #     # save execution data
    #     with open(f"{output_path}{dataset_name}_{synthesizer_name}_quality_report.json", "w") as json_file:
    #         json.dump(quality_report_obj, json_file)

    # if generate_sdv_diagnostic_report:
    #     print("Generating Diagnostic Report",  "#"*10)
    #     start_time = time.time()
    #     diagnostic_report_obj = compute_sdv_diagnostic_report(
    #         train_dataset,
    #         synthetic_dataset,
    #         metadata
    #     )
    #     execution_scores["diagnostic_report_time_sec"] = time.time() - \
    #         start_time
    #     print("Diagnostics: ", diagnostic_report_obj["results"])
    #     LOGGER.info("Diagnostics: ", diagnostic_report_obj["results"])

    #     # save execution data
    #     with open(f"{output_path}{dataset_name}_{synthesizer_name}_diagnostic_report.json", "w") as json_file:
    #         json.dump(diagnostic_report_obj, json_file)

    # save execution data
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_execution_scores.json", "w") as json_file:
        json.dump(execution_scores, json_file)

    print("-"*100)
    print(f"{synthesizer_name.upper()} trained  on {dataset_name.upper()} dataset| {num_samples} sampled | Files saved!")
    print("-"*100)
