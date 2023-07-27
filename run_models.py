from commons.static_vals import VALID_DATA_MODALITIES, DataModalities

# -----------------------------------------------------------------------------
# configurarions
# -----------------------------------------------------------------------------
exp_data_modality = "tabular"  # valid values -- sequential, free_text
# cuda (bool or str):
#    If ``True``, use CUDA. If a ``str``, use the indicated device.
#    If ``False``, do not use cuda at all.
use_gpu = False
num_epochs = 2

test_local = True
base_data_path = "data"

assert exp_data_modality in VALID_DATA_MODALITIES

if exp_data_modality == DataModalities.TABULAR.value:
    from run_sdv_tabular import run_tabular_models

    run_tabular_models(base_data_path, num_epochs, use_gpu, test_local)
