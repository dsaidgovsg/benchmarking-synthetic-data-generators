import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import List, Tuple

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
