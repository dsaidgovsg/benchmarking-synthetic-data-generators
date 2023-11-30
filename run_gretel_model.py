import json
import logging
import os
import sys
import time
import tracemalloc
import warnings
from io import StringIO

from commons.static_vals import N_BYTES_IN_MB
# from synthesizers.gretel_synthetics.sequential.dgan.dgan import (DGAN,
#                                                                  DGANConfig)
# from synthesizers.gretel_synthetics.tabular.actgan.actgan_wrapper import ACTGAN

# time varying features, fixed attributes, categorical variables, a
# and works well with many time sequence examples to train on.


from gretel_synthetics.timeseries_dgan.dgan import DGAN
from gretel_synthetics.timeseries_dgan.config import DGANConfig

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
    data_modality = kwargs["exp_data_modality"]
    synthesizer_name = kwargs["exp_synthesizer"]
    dataset_name = kwargs["exp_data"]
    use_gpu = kwargs["use_gpu"]
    num_epochs = kwargs["num_epochs"]
    output_path = kwargs["output_path"]
    train_dataset = kwargs["train_dataset"]
    # metadata = kwargs["metadata"]
    # num_samples=kwargs["num_samples"]

    if kwargs["sequential_details"]:
        num_sequences = kwargs["sequential_details"]["num_sequences"]
        max_sequence_length = kwargs["sequential_details"]["max_sequence_length"]
        seq_static_attributes = kwargs["sequential_details"]["static_attributes"]
        seq_dynamic_attributes = kwargs["sequential_details"]["dynamic_attributes"]
        time_attribute = kwargs["sequential_details"]["time_attribute"]
        entity = kwargs["sequential_details"]["entity"]
        discrete_attributes = kwargs["sequential_details"]["discrete_attributes"]

    # synthesizer_class = SYNTHESIZERS_MAPPING[synthesizer_name]
    num_samples = len(train_dataset)

    # TODO: Might need to save metadata JSON file for some datasets
    # metadata = detect_metadata(real_data)
    # LOGGER.info(metadata)

    tracemalloc.start()

    if synthesizer_name == "dgan":  # sequential
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
            # sample_len : max_sequence_length time series steps to generate from each LSTM cell in DGAN,
            # must be a divisor of max_sequence_len

            # TODO: remove 1, 2, 3, 4, 6, 7, 9, 12, 14, 18, 21, 28, 36, 42, 63, 84, 126, 252
            sample_len=max_sequence_length,
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

        # attribute_columns = seq_static_attributes
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
        # captured_print_out = StringIO()
        # sys.stdout = captured_print_out

        begin_train_time = time.time()
        synthesizer.train_dataframe(train_dataset,
                                    attribute_columns=seq_static_attributes,
                                    # feature_columns=seq_dynamic_attributes,
                                    example_id_column=entity,
                                    time_column=time_attribute,
                                    # discrete_columns=discrete_attributes,
                                    df_style="long")
        end_train_time = time.time()

        # Store the print statements in a variable
        # captured_print_out = StringsklearnIO()
        # sys.stdout = captured_print_out

        # ---------------------
        # Sample
        # ---------------------
        # Params:
        # num_sequences: An integer >0 describing the number of sequences to sample
        begin_sampling_time = time.time()
        synthetic_dataset = synthesizer.generate_dataframe(num_sequences)
        end_sampling_time = time.time()

    elif synthesizer_name == "actgan":  # sequential
        from gretel_synthetics.actgan.actgan_wrapper import ACTGAN
        """
            # https://synthetics.docs.gretel.ai/en/stable/models/actgan.html
        """
        print("Training with ACTGAN")
        synthesizer = ACTGAN(
            verbose=True,
            cuda=use_gpu,
            epochs=num_epochs,
            enforce_min_max_values=True, 
            # ---------------------
            # attributes for the future purposes
            # ---------------------
            # field_names = Optional[List[str]] = None,
            # field_types = Optional[Dict[str, dict]] = None,
            # field_transformers = Optional[Dict[str, Union[BaseTransformer, str]]] = None,
            # auto_transform_datetimes = bool = False,
            # anonymize_fields = Optional[Dict[str, str]] = None,
            # primary_key =  Optional[str] = None,
            # constraints: Optional[Union[List[Constraint], List[dict]]] = None,
            # table_metadata: Optional[Union[Metadata, dict]] = None,
            # embedding_dim: int = 128,
            # generator_dim: Sequence[int] = (256, 256),
            # discriminator_dim: Sequence[int] = (256, 256),
            # generator_lr: float = 2e-4,
            # generator_decay: float = 1e-6,
            # discriminator_lr: float = 2e-4,
            # discriminator_decay: float = 1e-6,
            # batch_size: int = 500,
            # discriminator_steps: int = 1,
            # binary_encoder_cutoff: int = 500,
            # binary_encoder_nan_handler: Optional[str] = None,
            # cbn_sample_size: Optional[int] = 250_000,
            # log_frequency: bool = True,
            # verbose: bool = False,
            # epochs: int = 300,
            # epoch_callback: Optional[Callable[[EpochInfo], None]] = None,
            # pac: int = 10,
            # learn_rounding_scheme: bool = True,
            # enforce_min_max_values: bool = True,
            # conditional_vector_type: ConditionalVectorType = ConditionalVectorType.SINGLE_DISCRETE,
            # conditional_select_mean_columns: Optional[float] = None,
            # conditional_select_column_prob: Optional[float] = None,
            # reconstruction_loss_coef: float = 1.0,
            # force_conditioning: bool = False,
            # conditional_vector_type="ANYWAY"
            # binary_encoder_cutoff=10, # use a binary encoder for data transforms if the cardinality of a column is below this value
        )
        # epoch_callback=epoch_tracker.add)

        # ---------------------
        # Train
        # ---------------------
        #  Store the print statements in a variable
        # captured_print_out = StringIO()
        # sys.stdout = captured_print_out

        begin_train_time = time.time()
        synthesizer.fit(train_dataset)
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
        synthetic_dataset = synthesizer.sample(num_samples)
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
    synthesizer.save(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pth")

    # Compute the size of the file in megabytes (MB)
    synthesizer_size = os.path.getsize(
        f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pth") / N_BYTES_IN_MB

    execution_scores = {
        # "Date": [today_date],
        "lib": f"GRETEL_0.20.0",
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
        f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv", index=False)

    # save execution data
    with open(f"{output_path}{dataset_name}_{synthesizer_name}_execution_scores.json", "w") as json_file:
        json.dump(execution_scores, json_file)

    print("-"*100)
    print(f"{synthesizer_name.upper()} trained  on {dataset_name.upper()} dataset| {num_samples} sampled | Files saved!")
    print("-"*100)
