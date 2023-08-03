import argparse
import json

from sdv.datasets.demo import download_demo

from commons.static_vals import VALID_DATA_MODALITIES, DataModalities
from synthesizers.sequential.gretel_synthetics.dgan import DGAN, DGANConfig

# time varying features, fixed attributes, categorical variables, a
# and works well with many time sequence examples to train on.

def run_model():

    config = DGANConfig(
        max_sequence_len=12,
        sample_len=12,
        # batch_size=8,
        epochs=2
    )

    model = DGAN(config)
    attribute_columns = ["Sector"]
    # feature_columns = ["Volume"]
    example_id_column  = "Symbol"
    time_column  = "Date"
    discrete_columns  = ["Sector"]
    df_style  = "long"

    model.train_dataframe(real_data,
                        attribute_columns=attribute_columns,
                        # feature_columns,
                        example_id_column=example_id_column,
                        time_column=time_column,
                        discrete_columns=discrete_columns,
                        df_style=df_style)
        
    syn_data = model.generate_dataframe(2)
    syn_data.to_csv("dgan.csv")
