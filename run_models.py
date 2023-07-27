import argparse
from commons.static_vals import VALID_DATA_MODALITIES, DataModalities


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--modality", "--m", type=str, default="tabular",
                        help="enter data modality. \
                        Possible values - {tabular, sequential, text}")
    parser.add_argument("--use_gpu", "--gpu", type=bool, default=False,
                        help="whether to use GPU device(s)")
    parser.add_argument("--num_epochs", "--e", type=int, default=2)
    parser.add_argument("--data_folder", "--d", type=str, default="data")
    parser.add_argument("--output_folder", "--o", type=str, default="output")

    args = parser.parse_args()

    # -----------------------------------------------------------------------------
    # configurarions
    # -----------------------------------------------------------------------------
    exp_data_modality = args.modality
    use_gpu = args.use_gpu
    num_epochs = args.num_epochs
    # test_local=args.test_local
    data_folder = args.data_folder
    output_folder = args.output_folder

    assert exp_data_modality in VALID_DATA_MODALITIES

    if exp_data_modality == DataModalities.TABULAR.value:
        from run_sdv_tabular import run_tabular_models

        run_tabular_models(num_epochs,
                           use_gpu, data_folder, output_folder)
