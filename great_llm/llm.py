import json
import os
# from sklearn.datasets import fetch_california_housing
import time
import tracemalloc

import pandas as pd
from be_great import GReaT
from sklearn.model_selection import train_test_split


def stratified_split_dataframe(df, target_column, return_only_train_data=True, test_size=0.2, random_state=42):
    """
    Perform stratified splitting of a pandas DataFrame.

    Args:
    - df (pandas.DataFrame): The input DataFrame.
    - target_column (str): The name of the target column containing class labels.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int or None): Seed for the random number generator.

    Returns:
    - X_train (pandas.DataFrame): Training features.
    - X_test (pandas.DataFrame): Test features.
    - y_train (pandas.Series): Training target.
    - y_test (pandas.Series): Test target.
    """
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    if return_only_train_data:
        return (X_train, y_train)
    else:
        return (X_train, y_train), (X_test, y_test)


def shuffle_and_split_dataframe(df, return_only_train_data=True, test_size=0.2, random_state=42):
    """
    Shuffle and split a pandas DataFrame into training and testing subsets.

    Args:
    - df (pandas.DataFrame): The input DataFrame.
    - test_size (float): The proportion of the dataset to include in the test split.
    - random_state (int or None): Seed for the random number generator.

    Returns:
    - X_train (pandas.DataFrame): Training features.
    - X_test (pandas.DataFrame): Test features.
    """
    # Shuffle the DataFrame
    shuffled_df = df.sample(
        frac=1, random_state=random_state).reset_index(drop=True)

    X = shuffled_df

    X_train, X_test = train_test_split(
        X, test_size=test_size, random_state=random_state
    )

    if return_only_train_data:
        return X_train
    else:
        return (X_train, X_test)


if __name__ == "__main__":

    epochs = 400
    dataset_name = "credit"  # health_insurance: 100, loan: 100, adult: 300, credit: 400
    output_path = "outputs"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    results = {}
    results["epochs"] = epochs

    ML_TASKS_TARGET_CLASS = {
        "adult": "label",
        "credit": "label",
        "loan": "Personal Loan",
        "health_insurance": "charges",
    }
    N_BYTES_IN_MB = 1000 * 1000

    real_dataset = pd.read_csv(f"data/{dataset_name}.csv")
    results["dataset_name"] = dataset_name
    results["num_rows"] = real_dataset.shape[0]
    results["num_cols"] = real_dataset.shape[1]

    if dataset_name in ["loan", "adult", "credit"]:
        (X_train, y_train) = stratified_split_dataframe(real_dataset,
                                                        ML_TASKS_TARGET_CLASS[dataset_name],
                                                        True)
        # Merge X_train and y_train columns horizontally
        train_dataset = pd.concat([X_train, y_train], axis=1)
    else:
        train_dataset = shuffle_and_split_dataframe(real_dataset, True)

    num_samples = len(real_dataset)
    results["num_samples"] = num_samples
    results["train_num_rows"] = train_dataset.shape[0]

    tracemalloc.start()
    try:
        model = GReaT(llm='distilgpt2', batch_size=32, epochs=epochs)

        train_time = time.time()
        model.fit(train_dataset)
        results["train_time"] = time.time() - train_time

        peak_memory = tracemalloc.get_traced_memory()[1] / N_BYTES_IN_MB
        tracemalloc.stop()
        tracemalloc.clear_traces()

        results["peak_memory_mb"] = peak_memory

        sample_time = time.time()
        synthetic_dataset = model.sample(n_samples=num_samples)
        results["sample_time"] = time.time() - sample_time
    except Exception as e:
        print("Exception:=> ", e)

    try:
        synthetic_dataset.to_csv(
            f"{output_path}/{dataset_name}_synthetic_data.csv", index=False)
    except Exception as e:
        print("Exception:=> ", e)

    try:
        # save execution data
        with open(f"{output_path}/report.json", "w") as json_file:
            json.dump(results, json_file)
    except Exception as e:
        print("Exception:=> ", e)

    try:
        # torch.save(self.model.state_dict(), path + "/model.pt")
        model.save(output_path)  # no /
    except Exception as e:
        print("Exception:=> ", e)


# print(type(data), data.shape)
# # print("fetch data")
# # model = GReaT(llm='distilgpt2', batch_size=32, epochs=1)
# # print("model traning")
# # model.fit(data)
# # print("model trained")
# # synthetic_data = model.sample(n_samples=10)
# # print("sampled")
