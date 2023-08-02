import argparse
import json

from sdv.datasets.demo import download_demo

from commons.static_vals import VALID_DATA_MODALITIES, DataModalities



# time varying features, fixed attributes, categorical variables, a
# and works well with many time sequence examples to train on.



real_data, metadata = download_demo(
    modality='sequential', 
    dataset_name='nasdaq100_2019'
)

real_data.to_csv("nasdaq100_2019.csv")

print("~"*10)
print(metadata)
print("~"*10)
json.dumps(metadata)

print(type(real_data), len(real_data))

print("done")

def run_model():
    ...

    #     # import numpy as np
    #     import pandas as pd
    #     from synthesizers.sequential.gretel_synthetics.dgan import DGAN
    #     from synthesizers.sequential.gretel_synthetics.dgan import DGANConfig

    #     # attributes = np.random.rand(10, 3)
    #     # features = np.random.rand(10, 20, 2)

    #     config = DGANConfig(
    #         max_sequence_len=12,
    #         sample_len=12,
    #         # batch_size=8,
    #         epochs=2
    #     )

    #     model = DGAN(config)
    #     attribute_columns = ["Sector"]
    #     # feature_columns = ["Volume"]
    #     example_id_column  = "Symbol"
    #     time_column  = "Date"
    #     discrete_columns  = ["Sector"]
    #     df_style  = "long"

    #     # print("---->", real_data.column)

    #     real_data = pd.read_csv("data/sequential/nasdaq100_2019.csv")

    #     model.train_dataframe(real_data,
    #                         attribute_columns=attribute_columns,
    #                         # feature_columns,
    #                         example_id_column=example_id_column,
    #                         time_column=time_column,
    #                         discrete_columns=discrete_columns,
    #                         df_style=df_style)
        
    #     print("Trained")
    #     syn_data = model.generate_dataframe(2)

    #     syn_data.to_csv("dgan.csv")
    #     print("Sampled")
