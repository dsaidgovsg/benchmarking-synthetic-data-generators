import argparse
import logging
import os
from datetime import datetime

from commons.static_vals import DEFAULT_EPOCH_VALUES
from commons.utils import get_dataset_with_sdv

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", "--l", type=str, default="sdv",
                        help="enter library. \
                        Possible values - {sdv, gretel, ydata}")
    parser.add_argument("--modality", "--m", type=str, default="tabular",
                        help="enter dataset modality. \
                        Possible values - {tabular, sequential, text}")
    parser.add_argument("--synthesizer", "--s", type=str, default="ctgan",
                        help="enter synthesizer name \
                            Possible values - {ctgan, tvae, gaussian_copula, par}")
    parser.add_argument("--data", type=str, default="adult",
                        help="enter dataset name.")
    
    parser.add_argument("--use_gpu", "--gpu", type=bool, default=False,
                        help="whether to use GPU device(s)")
    # deafult epoch is set as 0 as statistical models do not need epochs
    parser.add_argument("--num_epochs", "--e", type=int, default=0)
    parser.add_argument("--data_folder", "--d", type=str, default="data")
    parser.add_argument("--output_folder", "--o", type=str, default="output")


    # -----------------------------------------------------------
    # parsing inputs
    # -----------------------------------------------------------
    args = parser.parse_args()

    exp_library: str = args.library
    exp_data_modality: str = args.modality
    exp_synthesizer: str = args.synthesizer
    exp_dataset_name: str = args.data

    use_gpu: bool = args.use_gpu
    num_epochs: int = args.num_epochs
    data_folder: str = args.data_folder
    output_folder: str = args.output_folder

    # TODO: Add assertions to validate the inputs
    # assert exp_data_modality in VALID_DATA_MODALITIES
    # assert exp_synthesizer in VALID_MODALITY_SYNTHESIZER 
    # assert exp_data in VALID_MODALITY_DATA

    today_date = datetime.now().strftime("%Y-%m-%d")
    # -----------------------------------------------------------
    # Output folder for storing:
    # 1.Logs 2.Synthetic Data 3.Execution Metrics 4.Saved Models   
    # -----------------------------------------------------------
    output_path = f"{output_folder}/{today_date}/{exp_library}/{exp_data_modality}/{exp_synthesizer}/{exp_dataset_name}/"

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    logging.basicConfig(filename=f"{output_path}{exp_dataset_name}_{exp_synthesizer}.log",
                    format="%(asctime)s [%(filename)s:%(lineno)d] %(message)s ",
                    datefmt="%Y-%m-%d:%H:%M:%S",
                    filemode="w")
    LOGGER = logging.getLogger(__name__)
    LOGGER.setLevel(logging.DEBUG)

    # --------------
    # Get dataset
    # --------------
    sequential_details = None
    if exp_dataset_name == "adult":
        real_dataset, metadata = get_dataset_with_sdv("single_table", "adult")
    elif exp_dataset_name == "nasdaq":
        real_dataset, metadata = get_dataset_with_sdv("sequential", "nasdaq100_2019")
        sequential_details = {
            "num_sequences": 103,
            "max_sequence_length": 252,
            "fixed_attributes": ["Sector", "Industry"] # context_columns
        }
    elif exp_dataset_name == "census":
        real_dataset, metadata = get_dataset_with_sdv("single_table", "census")
    else: 
        # @TODO
        # real_dataset = load_dataset_df("")
        # dataset_path = f"{data_folder}/{data_modality}/{dataset_name}/{dataset_name}.csv"
        # data = pd.read_csv(dataset_path)
        ...

    # --------------
    # Run models
    # --------------
    if exp_library == "sdv":
        from run_sdv_model import run_model

        if not num_epochs:
            num_epochs = DEFAULT_EPOCH_VALUES["sdv"][exp_synthesizer]

        print("Selected Synthesizer Library: SDV")
        LOGGER.info((f"Modality: {exp_data_modality} | Synthesizer: {exp_synthesizer} | Dataset: {exp_dataset_name} | Epochs: {num_epochs}"))

        run_model(
            exp_data_modality=exp_data_modality,
            exp_synthesizer=exp_synthesizer,
            exp_data=exp_dataset_name,
            use_gpu=use_gpu,
            num_epochs=num_epochs,
            real_dataset=real_dataset,
            metadata=metadata,
            output_path=output_path,
            sequential_details=sequential_details)


    elif exp_library == "gretel":
        from run_gretel_model import run_model
        
        print("Selected Synthesizer Library: Gretel")
        # run_model(
        #     exp_data_modality=exp_data_modality,
        #     exp_synthesizer=exp_synthesizer,
        #     exp_data=exp_dataset_name,
        #     use_gpu=use_gpu,
        #     num_epochs=num_epochs,
        #     real_dataset=real_dataset,
        #     metadata=metadata,
        #     data_folder=data_folder,
        #     output_path=output_path,
        #     sequential_details=sequential_details)
