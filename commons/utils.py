"Utility functions"
from sdv.datasets.demo import download_demo
from sdv.metadata import SingleTableMetadata

# def get_execution_scores_obj():
#     return {
#         "Synthesizer": [],
#         "Dataset": [],
#         # "Dataset_Size_MB": [],
#         "Train_Time": [],
#         "Peak_Memory_MB": [],
#         "Synthesizer_Size_MB": [],
#         "Sample_Time": [],
#         "Device": []
#         # "Evaluate_Time": [],
#     }


def detect_metadata_with_sdv(real_data_df):
    """
    """
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data_df)
    # pprint(metadata.to_dict())
#     python_dict = metadata.to_dict()
    return metadata


def get_dataset_with_sdv(modality, dataset_name):
    """
    ------------------------
    single-table: 
    ------------------------
    >> from sdv.datasets.demo import get_available_demos 
    >> get_available_demos(modality='single_table')
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
    >> get_available_demos(modality='sequential')
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
    real_data, metadata = download_demo(
        modality=modality,
        dataset_name=dataset_name,
    )
    return real_data, metadata
