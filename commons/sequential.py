import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple


import pandas as pd
import numpy as np


def process_groups(df, required_group_size, group_by_column, operation='padding_and_truncate'):
    """
    Processes groups in a DataFrame based on the specified operation.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    required_group_size (int): The target size for each group.
    group_by_column (str): The name of the column to group by.
    operation (str): Operation to apply ('padding', 'truncate', 'padding_and_truncate', 'drop_and_truncate').

    Returns:
    pd.DataFrame: A new DataFrame with processed groups.
    """

    def pad_group(group, size, pad_value=0):
        rows_to_add = size - len(group)
        if rows_to_add > 0:
            padding_data = {col: [pad_value] * rows_to_add if col != group_by_column else [
                group[group_by_column].iloc[0]] * rows_to_add for col in group.columns}
            padding_df = pd.DataFrame(padding_data)
            return pd.concat([group, padding_df], ignore_index=True)
        return group

    def truncate_group(group, size):
        return group.head(size)

    # Process each group
    processed_groups = []
    for _, group in df.groupby(group_by_column):
        if operation == 'padding' and len(group) < required_group_size:
            group = pad_group(group, required_group_size)
        elif operation == 'truncate' and len(group) > required_group_size:
            group = truncate_group(group, required_group_size)
        elif operation == 'padding_and_truncate':
            if len(group) < required_group_size:
                group = pad_group(group, required_group_size)
            if len(group) > required_group_size:
                group = truncate_group(group, required_group_size)
        elif operation == 'drop_and_truncate':
            if len(group) >= required_group_size:
                group = truncate_group(group, required_group_size)
            else:
                continue  # Skip groups smaller than required size
        processed_groups.append(group)

    # Combine processed groups into a single DataFrame
    return pd.concat(processed_groups, ignore_index=True)


def get_groups_stats(df, column_to_groupby):
    """
    Calculates statistics of group sizes for a given DataFrame and column.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.
    column_to_groupby (str): The column name used for grouping the DataFrame.

    Returns:
    dict: A dictionary containing statistics of the group sizes.
    """

    # Grouping the DataFrame by the specified column and calculating group sizes
    group_sizes = df.groupby(column_to_groupby).size()

    # Calculating various metrics related to group sizes
    number_of_groups = group_sizes.count()  # Count number of groups
    max_group_size = group_sizes.max()      # Maximum group size
    min_group_size = group_sizes.min()      # Minimum group size
    average_group_size = group_sizes.mean()  # Average group size
    std_group_size = group_sizes.std()      # Standard deviation of group sizes

    # Print results
    print(f"Number of Groups: {number_of_groups}")
    print(f"Max Group Size: {max_group_size}")
    print(f"Min Group Size: {min_group_size}")
    print(f"Average Group Size: {average_group_size}")
    print(f"Standard deviation Group Size: {std_group_size}")

    # Creating a dictionary of calculated metrics
    metrics = {
        "Number of Groups": number_of_groups,
        "Max Group Size": max_group_size,
        "Min Group Size": min_group_size,
        "Average Group Size": average_group_size,
        "Standard Deviation of Group Sizes": std_group_size
    }

    return metrics


# TODO: this code needs some logic improvisation
def sequential_data_loader(
    data: pd.DataFrame,
    # seq_len: int = 10,
    id_col: str = 'id',
    target_col: str = 'target',
    time_col: str = 'time',
    static_cols: List[str] = None,
    temporal_cols: List[str] = None
) -> Tuple[pd.DataFrame, List[pd.DataFrame], List, pd.DataFrame]:
    """
    Generic function for loading sequential data with DataFrame output.
    Handles categorical encoding for processing.

    Parameters:
    - data (pd.DataFrame): The dataset to process.
    - seq_len (int): Length of the sequence.
    - id_col (str): Column name for identifier.
    - target_col (str): Column name of the target variable.
    - time_col (str): Column name for time.
    - static_cols (List[str]): List of static feature column names.
    - temporal_cols (List[str]): List of temporal feature column names.

    Returns:
    -------
    - static_data (pd.DataFrame): 
        A DataFrame containing static features for each entity. Each row corresponds to a unique entity, 
        and columns represent static attributes that do not change over time.

    - time_varying_data (List[pd.DataFrame]): 
        A list of DataFrames, each representing temporal data for an individual entity. 
        Each DataFrame in the list contains a sequence of observations over time, 
        with columns representing time-varying features.

    - observation_times (List): 
        A list of arrays, each containing the time points corresponding to each observation 
        in the temporal data for each entity.

    - outcome_data (pd.DataFrame): 
        A DataFrame containing the outcome variable of interest for each entity. 
        The structure and content of this DataFrame will depend on the nature of the target variable 
        (e.g., continuous, categorical, event occurrence).

    """

    # Handle default column selections
    if static_cols is None:
        static_cols = []
    if temporal_cols is None:
        temporal_cols = list(set(data.columns) -
                             set(static_cols) - {id_col, target_col, time_col})

    # Encode categorical columns
    label_encoders = {}
    for col in data.select_dtypes(include=['object', 'category']).columns:
        label_encoders[col] = LabelEncoder()
        data[col] = label_encoders[col].fit_transform(data[col])

    # Scale temporal features
    scaler = StandardScaler()
    data[temporal_cols] = scaler.fit_transform(data[temporal_cols])

    # Prepare outputs
    x_static, time_varying_data, observation_times, outcome = [], [], [], []

    for id_ in sorted(data[id_col].unique()):
        record_data = data[data[id_col] == id_]

        entity_static = record_data[static_cols].iloc[0]
        x_static.append(entity_static.tolist())

        entity_temporal = record_data[temporal_cols]
        time_varying_data.append(entity_temporal)

        times = record_data[time_col].values
        events = record_data[target_col].values

        observation_times.append(times)
        outcome.extend(events)

    static_data = pd.DataFrame(x_static, columns=static_cols)

    outcome_data = pd.DataFrame(outcome, columns=[target_col])

    return static_data, time_varying_data, observation_times, outcome_data
