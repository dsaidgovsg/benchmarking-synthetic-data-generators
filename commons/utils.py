"Utility functions"

from sdv.metadata import SingleTableMetadata

# TODO: add more info about the data


def get_execution_scores_obj():
    return {
        "Synthesizer": [],
        "Dataset": [],
        # "Dataset_Size_MB": [],
        "Train_Time": [],
        "Peak_Memory_MB": [],
        "Synthesizer_Size_MB": [],
        "Sample_Time": [],
        "Device": []
        # "Evaluate_Time": [],
    }


def detect_metadata(real_data_df):
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(data=real_data_df)
    # pprint(metadata.to_dict())
#     python_dict = metadata.to_dict()
    return metadata
