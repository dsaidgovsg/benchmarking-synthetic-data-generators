import logging
import os
import pickle
import time
import tracemalloc
import warnings

import pandas as pd

from commons.static_vals import (EXP_DATASETS, EXP_SYNTHESIZERS, N_BYTES_IN_MB,
                                 DataModalities)
from commons.utils import detect_metadata, get_execution_scores_obj
from synthesizers.tabular.sdv.synthesizer import (CTGANSynthesizer,
                                                  TVAESynthesizer)

# from sdv.evaluation.single_table import evaluate_quality, run_diagnostic


# from sdv.single_table import TVAESynthesizer
# from sdv.single_table import CTGANSynthesizer
# from sdv.single_table import GaussianCopulaSynthesizer
# from sdv.sequential import PARSynthesizer

with warnings.catch_warnings():
    warnings.simplefilter("ignore")


LOGGER = logging.getLogger(__name__)

SYNTHESIZER_MAPPING = {
    "ctgan": CTGANSynthesizer,
    "tvae": TVAESynthesizer
    # "gaussian_copula": GaussianCopulaSynthesizer
}

tabular = DataModalities.TABULAR.value


def run_tabular_models(num_epochs, use_gpu, data_folder, output_folder):
    synthesizers = EXP_SYNTHESIZERS[tabular]
    datasets = EXP_DATASETS[tabular]
    execution_scores = get_execution_scores_obj()

    for synthesizer_name in synthesizers:
        synthesizer_class = SYNTHESIZER_MAPPING[synthesizer_name]

        for dataset_name in datasets:

            dataset_path = f"{data_folder}/{tabular}/{dataset_name}/{dataset_name}.csv"

            data = pd.read_csv(dataset_path)
            real_data = data.copy()
            num_samples = len(real_data)

            metadata = detect_metadata(real_data)

            tracemalloc.start()

            synthesizer = synthesizer_class(
                metadata, epochs=num_epochs, cuda=use_gpu)

            begin_time = time.time()  # datetime.utcnow()
            # ---------------------
            # Train
            # ---------------------
            synthesizer.fit(real_data)
            train_time = time.time()
            # ---------------------
            # Sample
            # ---------------------
            synthetic_data = synthesizer.sample(num_rows=num_samples)
            sampling_time = time.time()

            peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
            tracemalloc.stop()
            tracemalloc.clear_traces()

            # TODO: replace with logs
            print(
                f"Model {synthesizer_name} trained on {dataset_name} and sampled.")

            synthesizer_size = len(pickle.dumps(synthesizer)) / N_BYTES_IN_MB

            execution_scores["Synthesizer"].append(synthesizer_name)
            execution_scores["Dataset"].append(dataset_name)
            execution_scores["Train_Time"].append(train_time - begin_time)
            execution_scores["Peak_Memory_MB"].append(peak_memory)
            execution_scores["Synthesizer_Size_MB"].append(synthesizer_size)
            execution_scores["Sample_Time"].append(sampling_time - train_time)
            execution_scores["Device"].append("GPU" if use_gpu else "CPU")

            output_path = f"{output_folder}/{tabular}/{synthesizer_name}/{dataset_name}/"

            if not os.path.exists(output_path):
                os.makedirs(output_path)

            # save model
            synthesizer.save(
                filepath=f"{output_path}{dataset_name}_{synthesizer_name}_synthesizer.pkl"
            )

            # save synthetic data
            synthetic_data.to_csv(
                f"{output_path}{dataset_name}_{synthesizer_name}_synthetic_data.csv")

    execution_scores_df = pd.DataFrame(execution_scores)
    execution_scores_df.to_csv(
        f"{output_folder}/{tabular}/execution_scores.csv")


#     quality_report_obj = evaluate_quality(
#         real_data,
#         synthetic_data,
#         metadata
#     )
#     print("quality_report_obj: ", quality_report_obj)

#     diagnostic_report_obj = run_diagnostic(
#         real_data,
#         synthetic_data,
#         metadata
#     )
#     print("diagnostic_report_obj: ", diagnostic_report_obj)
