"Utility functions"
import numpy as np
import pandas as pd
from scipy import stats
import sdv

# from sdv.datasets.demo import download_demo
# from sdv.metadata import SingleTableMetadata
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KernelDensity

# https://docs.sdv.dev/sdmetrics/metrics/metrics-in-beta/outliercoverage


def detect_metadata_with_sdv(real_data_df: pd.DataFrame):
    """
    Automatically detect the metadata based on your actual data using SDV API.
    Args:
        real_data_df: pandas.DataFrame 
    Returns: 
        metadata: SingleTableMetadata
    """
    metadata = sdv.metadata.SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data_df)
#     python_dict = metadata.to_dict()
    return metadata


def get_dataset_with_sdv(modality: str, dataset_name: str):
    """
    Get dataset from the SDV's public repository.
    Args:
        modality: valid values {"single_table", "sequential"}
        dataset_name: valid datasets are listed below
    Returns:
        real_data: pandas.Dataframe
        metadata: SingleTableMetadata
    ------------------------
    single-table: 
    ------------------------
    >> from sdv.datasets.demo import get_available_demos 
    >> get_available_demos(modality="single_table")
                        dataset_name  size_MB num_tables
        0                   KRK_v1     0.07          1
        1                    adult     3.91          1
        2                    alarm     4.52          1
        3                     asia     1.28          1
        4                   census    98.17          1
        5          census_extended     4.95          1
        6                    child     3.20          1
        7                  covtype   255.65          1
        8                   credit    68.35          1
        9       expedia_hotel_logs     0.20          1
        10          fake_companies     0.00          1
        11       fake_hotel_guests     0.03          1
        12                    grid     0.32          1
        13                   gridr     0.32          1
        14               insurance     3.34          1
        15               intrusion   162.04          1
        16                 mnist12    81.20          1
        17                 mnist28   439.60          1
        18                    news    18.71          1
        19                    ring     0.32          1
        20      student_placements     0.03          1
        21  student_placements_pii     0.03          1

    ------------------------
    sequential:
    ------------------------
    >> from sdv.datasets.demo import get_available_demos 
    >> get_available_demos(modality=sequential")
                                dataset_name  size_MB num_tables
            0   ArticularyWordRecognition     8.61          1
            1          AtrialFibrillation     0.92          1
            2                BasicMotions     0.64          1
            3       CharacterTrajectories    19.19          1
            4                     Cricket    17.24          1
            5               DuckDuckGeese   291.38          1
            6                       ERing     1.25          1
            7                  EchoNASDAQ     4.16          1
            8                  EigenWorms   372.63          1
            9                    Epilepsy     3.17          1
            10       EthanolConcentration    51.38          1
            11              FaceDetection   691.06          1
            12            FingerMovements     5.32          1
            13      HandMovementDirection    10.48          1
            14                Handwriting     8.51          1
            15                  Heartbeat    86.14          1
            16             InsectWingbeat   547.59          1
            17             JapaneseVowels     1.28          1
            18                       LSST    14.18          1
            19                     Libras     0.78          1
            20               MotorImagery   616.90          1
            21                     NATOPS     4.11          1
            22                    PEMS-SF   490.15          1
            23                  PenDigits     4.22          1
            24             PhonemeSpectra   173.63          1
            25               RacketSports     0.73          1
            26         SelfRegulationSCP1    40.21          1
            27         SelfRegulationSCP2    38.52          1
            28         SpokenArabicDigits    47.63          1
            29              StandWalkJump     4.32          1
            30        UWaveGestureLibrary     7.76          1
            31             nasdaq100_2019     1.65          1
    """
    # try:
    from sdv.datasets.demo import download_demo

    real_data, metadata = download_demo(
        modality=modality,
        dataset_name=dataset_name,
    )
    return real_data, metadata
    # except:
    #     return


def detect_distribution(column):
    """
    Detects the distribution that best fits the data in a given column.

    Args:
    - column (pandas.Series): The input column containing numerical data.

    Returns:
    - detected_distribution (str): The name of the detected distribution.
    """

    # Removing NaN values from the column
    data = column.dropna()

    # Calculate basic statistics
    mean = data.mean()
    std = data.std()
    skewness = data.skew()

    # Fit different distributions and calculate goodness of fit scores
    distribution_scores = {
        'norm': np.abs(stats.norm.fit(data)[1] - std),
        'beta': np.abs(stats.beta.fit(data)[1] - std),
        'truncnorm': np.abs(stats.truncnorm.fit(data)[1] - std),
        'uniform': np.abs(stats.uniform.fit(data)[1] - std),
        'gamma': np.abs(stats.gamma.fit(data)[1] - std)
    }

    # Fit Gaussian KDE and calculate log-likelihood
    kde = KernelDensity(bandwidth=0.2).fit(data.values.reshape(-1, 1))
    log_likelihood = kde.score_samples(data.values.reshape(-1, 1)).sum()

    distribution_scores['gaussian_kde'] = -log_likelihood

    # Find the distribution with the smallest score
    detected_distribution = min(
        distribution_scores, key=distribution_scores.get)

    return detected_distribution


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
