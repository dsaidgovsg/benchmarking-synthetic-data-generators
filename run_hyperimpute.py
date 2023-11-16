""" To run hyperimpute on columns with null values"""
import tracemalloc
import time
import json
import pandas as pd

from hyperimpute.plugins.imputers import Imputers
from commons.static_vals import N_BYTES_IN_MB
from sklearn.preprocessing import OneHotEncoder


def _validate_imputers(imputers: Imputers, plugin_name: str) -> bool:
    """

    Args:
        impute (Imputers)
        plugin_name

    Returns:
        bool
    """
    list_of_imputers = imputers.list()

    if plugin_name in list_of_imputers:
        return True
    else:
        return False


def _validate_parameters(plugin_name):
    # can use pydantic to validate
    ...


def _encode(dataset, categorical_columns):

    encoder = OneHotEncoder(sparse=False, drop='first')

    # Apply one-hot encoding to categorical columns
    encoded_data = encoder.fit_transform(dataset[categorical_columns])
    encoded_df = pd.DataFrame(
        encoded_data, columns=encoder.get_feature_names_out(categorical_columns))

    encoded_dataset = pd.concat(
        [dataset.drop(columns=categorical_columns), encoded_df], axis=1)

    return encoded_dataset


def _ampute():
    # TODO: for simulation of inserting null data
    from hyperimpute.plugins.utils.simulate import simulate_nan
    ...


def hyperimpute(dataset: pd.DataFrame, dataset_name: str, plugin_name: str, output_path: str, **plugin_params) -> pd.DataFrame:
    """
    run imputation on dataset

    Args:
    - dataset: original dataset
    - plugin_name: Name of imputation plugin
    - output_path: path to write metrics
    - plugins_parameters: contains all the different parameters combinations in keyword arguments.

    Returns:
        pd.DataFrame: imputed dataset
    """
    # -----
    # prep
    # -----
    imputers = Imputers()

    # identify columns with missing values and without
    columns_with_missing_values = dataset.columns[dataset.isnull(
    ).any()].tolist()
    columns_without_missing_values = [
        col for col in dataset.columns if col not in columns_with_missing_values]

    # identify data types of columns
    numeric_columns = list(dataset.select_dtypes(
        include=['int64', 'float64']).columns)
    categorical_columns = list(
        dataset.select_dtypes(include=['category']).columns)
    # [ col for col in dataset.columns if dataset[col].dtype == 'category']

    # -----
    # validations
    # -----
    if not _validate_imputers(imputers,
                              plugin_name=plugin_name):
        raise ValueError(f"No such plugin, {plugin_name}")

    if len(columns_with_missing_values) < 1:
        print("No columns with null values found, skipping hyperimpute")
        return dataset

    _validate_parameters(plugin_name)

    # -----
    # main logic
    # -----

    tracemalloc.start()

    if plugin_name in ["mean", "median"]:
        plugin = imputers.get(plugin_name)

        dataset = _encode(dataset, categorical_columns)
        target_columns = numeric_columns + categorical_columns
        if not numeric_columns:
            print(
                f"No appropriate columns to impute with plugin, {plugin_name}")

    elif plugin_name == "most_frequent":
        plugin = imputers.get(plugin_name)
        target_columns = columns_with_missing_values

    elif plugin_name == "hyperimpute":
        plugin = imputers.get(plugin_name,
                              optimizer="hyperband",
                              classifier_seed=["logistic_regression"],
                              regression_seed=["linear_regression"])
        target_columns = columns_with_missing_values

    elif plugin_name == "mice":

        plugin = imputers.get(
            plugin_name,
            # n_imputations=plugin_params["n_imputations"],
            # max_iter=plugin_params["max_iter"],
            # random_state=plugin_params["random_state"]
        )

        dataset = _encode(dataset, categorical_columns)
        target_columns = numeric_columns + categorical_columns
        if not numeric_columns:
            print(
                f"No appropriate columns to impute with plugin, {plugin_name}")

    elif plugin_name == "missforest":
        plugin = imputers.get(
            plugin_name,
            # n_estimators=plugin_params["n_estimators"],
            # max_iter=plugin_params["max_iter"],
            # random_state=plugin_params["random_state"]
        )

    elif plugin_name == "ice":
        plugin = imputers.get(
            plugin_name,
            # max_iter=plugin_params["max_iter"],
            # random_state=plugin_params["random_state"]
        )

    else:
        raise NotImplementedError(f"No such hyperimpute plugin: {plugin_name}")

    # impute only on columns with missing values
    begin_impute_time = time.time()
    imputed_columns = plugin.fit_transform(
        dataset[target_columns]
    )

    # concatenate imputed dataframe
    imputed_df = pd.DataFrame(
        imputed_columns, columns=columns_with_missing_values, index=dataset.index)
    imputed_dataset = pd.concat(
        [dataset[columns_without_missing_values], imputed_df], axis=1)

    end_impute_time = time.time()
    peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
    tracemalloc.stop()
    tracemalloc.clear_traces()

    # ----
    # post-experiment
    # ----

    execution_scores = {
        "lib": "hyperimpute==0.1.17",
        "plugin": plugin_name,

        "impute_time_sec": end_impute_time - begin_impute_time,

        "peak_memory_mb": peak_memory,

    }

    # ---------------------
    # Dump output to files
    # ---------------------

    # save imputed data
    imputed_dataset.to_csv(
        f"{output_path}{dataset_name}_imputed_data.csv")

    # save execution data
    with open(f"{output_path}{dataset_name}_execution_scores.json", "w", encoding="utf-8") as json_file:
        json.dump(execution_scores, json_file)

    return imputed_dataset
