# import os
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
from synthesizers.sequential.sdv.par_synthesizer import PARSynthesizer
# SDV synthesizers 
from synthesizers.tabular.sdv.copulas_synthesizer import \
    GaussianCopulaSynthesizer
from synthesizers.tabular.sdv.gen_synthesizer import (CTGANSynthesizer,
                                                      TVAESynthesizer)

# from sdv.evaluation.single_table import evaluate_quality, run_diagnostic

# from sdv.single_table import TVAESynthesizer
# from sdv.single_table import CTGANSynthesizer
# from sdv.single_table import GaussianCopulaSynthesizer
# from sdv.sequential import PARSynthesizer

# Links to Transform and Anonymize
# https://docs.sdv.dev/sdv/single-table-data/modeling/synthetic-data-workflow/transform-and-anonymize

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# tabular = DataModalities.TABULAR.value

SYNTHESIZERS_MAPPING = {
    # The CTGAN Synthesizer uses GAN-based DL methods
    # deep learning methods to train a model and generate synthetic data.
    "ctgan": CTGANSynthesizer,
    # The TVAE Synthesizer uses a variational autoencoder (VAE)-based DL methods
    "tvae": TVAESynthesizer,
    # The Gaussian Copula Synthesizer uses classic,stattical methods
    "gaussian_copula": GaussianCopulaSynthesizer,
    # The PARSynthesizer uses a DL methods
    "par": PARSynthesizer
}

# synthesizers = EXP_SYNTHESIZERS[tabular]
# datasets = EXP_DATASETS[tabular]

# TODO:
# storing metadata -- load from the dict

def run_model(**kwargs):
    """"""
    data_modality=kwargs["exp_data_modality"]
    synthesizer_name=kwargs["exp_synthesizer"]
    dataset_name=kwargs["exp_data"]
    use_gpu=kwargs["use_gpu"]
    num_epochs=kwargs["num_epochs"]
    # data_folder=kwargs["data_folder"]
    output_path=kwargs["output_path"]
    real_dataset=kwargs["real_dataset"]
    metadata=kwargs["metadata"]
    # num_samples=kwargs["num_samples"]


    if kwargs["sequential_details"]:
        num_sequences = kwargs["sequential_details"]["num_sequences"] 
        max_sequence_length = kwargs["sequential_details"]["max_sequence_length"] 
        seq_fixed_attributes = kwargs["sequential_details"]["fixed_attributes"] 

    synthesizer_class = SYNTHESIZERS_MAPPING[synthesizer_name]
    num_samples = len(real_dataset)

    # TODO: Might need to save metadata JSON file for some datasets
    # metadata = detect_metadata(real_data)
    LOGGER.info(metadata)

    tracemalloc.start()

    if synthesizer_name == "gaussian_copula":
        # ---------------------
        # Gaussain Copula
        # ---------------------
        # Learns - (1) Marginal Distributions (2) Covariance
        # Link to the paper: https://dai.lids.mit.edu/wp-content/uploads/2018/03/SDV.pdf
        # Params:
        # numerical_distributions = {
        #     <column name>: 'norm',
        # },
        # One of: 'norm' 'beta', 'truncnorm', 'uniform', 'gamma' or 'gaussian_kde'
        # default_distribution <str> ='beta'
        synthesizer = synthesizer_class(metadata)
    elif synthesizer_name == "par": # sequential
        # ---------------------------------------------
        # PAR is a Probabilistic Auto-Regressive model 
        # ---------------------------------------------
        # - Learns how to create brand new sequences of multi-dimensional data, 
        #   by conditioning on the unchanging, context values.
        # - Models non-numerical columns, including columns with missing values.
        # - Designed to work on multi-sequence data
        # - Link to the paper: https://arxiv.org/pdf/2207.14406.pdf
        # Params:
        # context_columns <list>: do not vary inside of a sequence.    
        synthesizer = synthesizer_class(metadata,
                                        context_columns=seq_fixed_attributes,
                                        epochs=num_epochs,
                                        cuda=use_gpu, 
                                        verbose=True,)
        
    elif synthesizer_name == "ctgan":
        # --------------------------
        # Conditional Table GAN
        # --------------------------
        # Params:
        # epochs: The optimal number of epochs depends on both the complexity of your dataset 
        #         and the metrics you are using to quantify success.
        # links: https://github.com/sdv-dev/SDV/discussions/980
        #        https://datacebo.com/blog/interpreting-ctgan-progress/
        synthesizer = synthesizer_class(metadata, 
                                        epochs=num_epochs,
                                        cuda=use_gpu,
                                        verbose=True)
    elif synthesizer_name == "tvae":
        # --------------------------
        # TVAE
        # --------------------------
        synthesizer = synthesizer_class(metadata,
                                epochs=num_epochs,
                                cuda=use_gpu)

    LOGGER.info(synthesizer.get_parameters())

    captured_print_out = StringIO()
    sys.stdout = captured_print_out

    begin_time = time.time()
    # ---------------------
    # Train
    # ---------------------
    synthesizer.fit(real_dataset)
    
    if synthesizer_name == "gaussian_copula":
        LOGGER.info(synthesizer.get_learned_distributions())
    train_time = time.time()
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
        synthetic_dataset = synthesizer.sample(num_sequences=num_sequences,
                                            sequence_length=max_sequence_length)
    else:
        synthetic_dataset = synthesizer.sample(num_rows=num_samples)
    sampling_time = time.time()

    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()
    
    # ---------------------
    # Outputs
    # ---------------------
    synthesizer_size = len(pickle.dumps(synthesizer)) / N_BYTES_IN_MB

    # Get the memory usage of the real and synthetic DataFrame in bytes
    real_dataset_size = real_dataset.memory_usage(deep=True).sum() / N_BYTES_IN_MB
    synthetic_dataset_size = synthetic_dataset.memory_usage(deep=True).sum() / N_BYTES_IN_MB

    execution_scores = {
        # "Date": [today_date],
        "lib": f"SDV_{sdv.__version__}", 
        "modality": data_modality, 
        "synthesizer": synthesizer_name, 
        "dataset": dataset_name,
        "num_rows": real_dataset.shape[0], 
        "num_cols": real_dataset.shape[1], 
        "num_epochs": num_epochs if num_epochs else 0,
        # "num_cat": len(dataset_name), 
        # "num_numeric": len(dataset_name), 
        "train_time_sec": train_time - begin_time, 
        "sample_time_sec": sampling_time - train_time, 
        "peak_memory_mb": peak_memory,
        "synthesizer_size_mb": synthesizer_size,
        "real_dataset_size_mb": real_dataset_size,
        "synthetic_dataset_size_mb": synthetic_dataset_size,  
        "device": "GPU" if use_gpu else "CPU"
    }

    # execution_scores = get_execution_scores_obj()
    # execution_scores["Date"].append(today_date)
    # execution_scores["Lib"].append(f"SDV_{sdv.__version__}")
    # execution_scores["Modality"].append(data_modality)
    # execution_scores["Synthesizer"].append(synthesizer_name)
    # execution_scores["Dataset"].append(dataset_name)
    # # TODO:
    # # add dataset characterstics 
    # execution_scores["Train_Time"].append(train_time - begin_time)
    # execution_scores["Peak_Memory_MB"].append(peak_memory)
    # execution_scores["Synthesizer_Size_MB"].append(synthesizer_size)
    # execution_scores["Sample_Time"].append(sampling_time - train_time)
    # execution_scores["Device"].append("GPU" if use_gpu else "CPU")

    sys.stdout = sys.__stdout__
    captured_print_out = captured_print_out.getvalue()

    # save print statements 
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_out.txt", "w") as json_file:
        json.dump(captured_print_out, json_file)

    # save model
    synthesizer.save(
        filepath=f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pkl"
    )
    # save synthetic data
    synthetic_dataset.to_csv(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv")
  
    # save execution data
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_execution_scores.json", "w") as json_file:
        json.dump(execution_scores, json_file)


    print("-"*30)
    print(f"{synthesizer_name.upper()} trained  on {dataset_name.upper()} dataset| {num_samples} sampled | Files saved!")
    print("-"*30)


    # execution_scores_df = pd.DataFrame(execution_scores)
    # execution_scores_df.to_csv(
    #     f"{output_path}{dataset_name}_{synthesizer_name}_execution_scores.csv")

    # TODO: Time consuming! 
    # quality_report_obj = evaluate_quality(
    #     real_dataset,
    #     synthetic_dataset,
    #     metadata
    # )
    # print("quality_report_obj: ", quality_report_obj)

    # diagnostic_report_obj = run_diagnostic(
    #     real_dataset,
    #     synthetic_dataset,
    #     metadata
    # )
    # print("diagnostic_report_obj: ", diagnostic_report_obj)

# df = pd.read_csv("/Users/anshusingh/DPPCC/whitespace/benchmarking-synthetic-data-generators/data/sequential/nasdaq100_2019.csv")
# groups = df.groupby('Symbol').size().reset_index(name='Count')
# print(groups, min(list(groups["Count"])),  max(list(groups["Count"])))
# # for Symbol, Count in zip(list(groups["Symbol"]), list(groups["Count"])):
# #     print(Symbol, Count)
# breakpoint()
