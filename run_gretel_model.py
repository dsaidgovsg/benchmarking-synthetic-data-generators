import os 
import sys
import json
import logging
import time
import tracemalloc
import warnings
from io import StringIO

from commons.static_vals import N_BYTES_IN_MB
from synthesizers.gretel_synthetics.sequential.dgan.dgan import DGAN, DGANConfig
from synthesizers.gretel_synthetics.tabular.actgan.actgan_wrapper import ACTGAN


# time varying features, fixed attributes, categorical variables, a
# and works well with many time sequence examples to train on.

with warnings.catch_warnings():
    warnings.simplefilter("ignore")

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.DEBUG)

# tabular = DataModalities.TABULAR.value

# SYNTHESIZERS_MAPPING = {
#     # This model supports time varying features, fixed attributes, categorical variables, 
#     # and works well with many time sequence examples to train on
#     "dgan": DGAN,
#     # "actgan": ACTGAN
#     # "lstm": LSTM
# }

def run_model(**kwargs):
    """"""
    data_modality=kwargs["exp_data_modality"]
    synthesizer_name=kwargs["exp_synthesizer"]
    dataset_name=kwargs["exp_data"]
    use_gpu=kwargs["use_gpu"]
    num_epochs=kwargs["num_epochs"]
    output_path=kwargs["output_path"]
    real_dataset=kwargs["real_dataset"]
    metadata=kwargs["metadata"]
    # num_samples=kwargs["num_samples"]
    
    if kwargs["sequential_details"]:
        num_sequences = kwargs["sequential_details"]["num_sequences"]
        max_sequence_length = kwargs["sequential_details"]["max_sequence_length"]
        seq_fixed_attributes = kwargs["sequential_details"]["fixed_attributes"]
        seq_varying_attributes = kwargs["sequential_details"]["varying_attributes"]
        time_attribute = kwargs["sequential_details"]["time_attribute"]
        entity = kwargs["sequential_details"]["entity"]
        discrete_attributes = kwargs["sequential_details"]["discrete_attributes"]

    # synthesizer_class = SYNTHESIZERS_MAPPING[synthesizer_name]
    num_samples = len(real_dataset)

    # TODO: Might need to save metadata JSON file for some datasets
    # metadata = detect_metadata(real_data)
    LOGGER.info(metadata)

    tracemalloc.start()

    if synthesizer_name == "dgan": # sequential
        # ---------------------------------------------
        # DoppleGANger
        # ---------------------------------------------
        # DoppelGANger is a generative adversarial network (GAN) model for time series.
        # It supports multi-variate time series (referred to as features) and fixed variables for each time series (attributes)
        # The combination of attribute values and sequence of feature values is 1 example.
        # Once trained, the model can generate novel examples that exhibit the same temporal correlations 
        # as seen in the training data.
        # Link to the documnetation (listing the parameters)
        # https://synthetics.docs.gretel.ai/en/stable/models/timeseries_dgan.html#timeseries-dgan
        # https://docs.gretel.ai/reference/synthetics/models/gretel-dgan
        # - Link to the paper: https://arxiv.org/abs/1909.13403
        # Params:
        # max_sequence_len – length of time series sequences, variable length sequences are not supported,
        #                    so all training and generated data will have the same length sequences.
        config = DGANConfig(
                max_sequence_len=max_sequence_length,
                # sample_len : time series steps to generate from each LSTM cell in DGAN, 
                # must be a divisor of max_sequence_len

                # TODO: remove 1, 2, 3, 4, 6, 7, 9, 12, 14, 18, 21, 28, 36, 42, 63, 84, 126, 252
                sample_len=1,
                # apply_feature_scaling: bool = True,
                # apply_example_scaling: bool = True,
                # batch_size=1000,
                # use_attribute_discriminator: bool = True, 
                # discriminator_learning_rate: float = 0.001
                # generator_learning_rate: float = 0.001,
                # attribute_discriminator_learning_rate: float = 0.001,
                epochs=num_epochs,
                cuda=use_gpu)   

        LOGGER.info("DGAN Configuration")
        LOGGER.info(config.to_dict())

        # Interface for training model and generating data based on configuration in an DGANConfig instance.
        synthesizer = DGAN(config)

        # attribute_columns = seq_fixed_attributes
        # feature_columns = ["Open", "Close", "Volume", "MarketCap"]
        # example_id_column  = "Symbol"
        # time_column  = "Date"
        # discrete_columns  = ["Sector"]

        # - “Wide” format uses one row for each example with 0 or more attribute columns and 
        # 1 column per time point in the time series. “Wide” format is restricted to 1 feature variable. 
        # - “Long” format uses one row per time point, supports multiple feature variables, and uses
        # additional example id to split into examples and time column to sort.        
        # ---------------------
        # Train
        # ---------------------
        #  Store the print statements in a variable
        captured_print_out = StringIO()
        sys.stdout = captured_print_out

        begin_train_time = time.time()
        synthesizer.train_dataframe(real_dataset,
                    attribute_columns=seq_fixed_attributes,
                    feature_columns=seq_varying_attributes,
                    example_id_column=entity,
                    time_column=time_attribute,
                    discrete_columns=discrete_attributes,
                    df_style="long")
        end_train_time = time.time()
        
        # Store the print statements in a variable
        # captured_print_out = StringIO()
        # sys.stdout = captured_print_out

        # ---------------------
        # Sample
        # ---------------------
        # Params: 
        # num_sequences: An integer >0 describing the number of sequences to sample
        begin_sampling_time = time.time()
        synthetic_dataset = synthesizer.generate_dataframe(num_sequences)
        end_sampling_time = time.time()

    elif synthesizer_name == "actgan": # sequential 

        model = ACTGAN(
                    verbose=True,
                    # binary_encoder_cutoff=10, # use a binary encoder for data transforms if the cardinality of a column is below this value
                    epochs=2)
            # epoch_callback=epoch_tracker.add)   
  
        # ---------------------
        # Train
        # ---------------------
        #  Store the print statements in a variable
        captured_print_out = StringIO()
        sys.stdout = captured_print_out

        begin_train_time = time.time()
        model.fit(real_dataset)
        end_train_time = time.time()
        
        print("@"*100)
        breakpoint()
        # Store the print statements in a variable
        # captured_print_out = StringIO()
        # sys.stdout = captured_print_out

        # ---------------------
        # Sample
        # ---------------------
        # Params: 
        # num_sequences: An integer >0 describing the number of sequences to sample
        begin_sampling_time = time.time()
        synthetic_dataset = model.sample(num_samples)
        end_sampling_time = time.time()





    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()
    
    # ---------------------
    # Prepare Outputs
    # ---------------------

    sys.stdout = sys.__stdout__
    captured_print_out = captured_print_out.getvalue()
    print(captured_print_out)

    # ---------------------
    # Dump output to files
    # ---------------------
    # save print statements 
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_out.txt", "w") as log_file:
        json.dump(captured_print_out, log_file)

    # Get the memory usage of the real and synthetic dataFrame in MB
    real_dataset_size_deep = real_dataset.memory_usage(deep=True).sum() / N_BYTES_IN_MB
    synthetic_dataset_size_deep = synthetic_dataset.memory_usage(deep=True).sum() / N_BYTES_IN_MB

    real_dataset_size = real_dataset.memory_usage(deep=False).sum() / N_BYTES_IN_MB
    synthetic_dataset_size = synthetic_dataset.memory_usage(deep=False).sum() / N_BYTES_IN_MB

    # Save model as Pytorch checkpoint
    synthesizer.save(f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pth")

    # Compute the size of the file in megabytes (MB)
    synthesizer_size = os.path.getsize(f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pth") / N_BYTES_IN_MB

    execution_scores = {
        # "Date": [today_date],
        "lib": f"GRETEL_0.20.0", 
        "modality": data_modality, 
        "synthesizer": synthesizer_name, 
        
        "dataset": dataset_name,
        "num_rows": real_dataset.shape[0], 
        "num_cols": real_dataset.shape[1], 
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
        "real_dataset_size_mb_deep": real_dataset_size_deep,
        
        "synthetic_dataset_size_mb": synthetic_dataset_size,
        "real_dataset_size_mb": real_dataset_size
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
