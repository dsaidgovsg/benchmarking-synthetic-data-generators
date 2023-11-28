from typing import List
import time
import json
import optuna
from enum import Enum

from synthcity.plugins import Plugins
from synthcity.benchmark import Benchmarks
from synthcity.utils.optuna_sample import suggest_all
from synthcity.plugins.core.dataloader import GenericDataLoader
from commons.static_vals import (DEFAULT_EPOCH_VALUES,
                                 MLTasks,
                                 ML_REGRESSION_TASK_DATASETS,
                                 ML_CLASSIFICATION_TASK_DATASETS,
                                 ML_TASKS_TARGET_CLASS)



# New delta distribution


# Relevant links
# https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/benchmark/__init__.py
# https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/metrics/eval_statistical.py

# code 
# https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/utils/optuna_sample.py
# https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/plugins/core/distribution.py
# https://github.com/vanderschaarlab/synthcity/blob/main/src/synthcity/benchmark/__init__.py


class StudyDirection(Enum):
    """
    Represents the direction of optimization for an Optuna study.

    Attributes:
    - MINIMIZE: The study aims to minimize the objective function value.
    - MAXIMIZE: The study aims to maximize the objective function value.
    """

    MINIMIZE = "minimize"
    """Represents the direction where the study minimizes the objective function."""

    MAXIMIZE = "maximize"
    """Represents the direction where the study maximizes the objective function."""


# def objective(trial: optuna.Trial) -> float:
def objective(trial, synthsizer_name,
              train_dataset,
              test_dataset,
              objective_metrics: List[str],
              objective_direction_str,
              opt_dict,
              out_json_name) -> float:
    """
    Objective function for Optuna optimization.

    Parameters:
    - trial (optuna.Trial): The trial object.

    Returns:
    - score (float): The objective value to be minimized or maximized.
    """
    trial_start_time = time.time()
    # trial_num = trial.number
    trial_id = f"trial_{trial.number}"
    print(f"Trial ID  {trial_id}")
    opt_dict["trials"][trial_id] = {}
    trial_obj = opt_dict["trials"][trial_id]
    # Get the hyperparameter space from the specified synthesizer
    hp_space = Plugins().get(synthsizer_name).hyperparameter_space()

    # parameter delta must be in range 0 <= delta <= 0.5
    # the synthcity library has set range [0, 50]
    if synthsizer_name == "arf":
        from synthcity.plugins.core.distribution import FloatDistribution
        new_delta_distribution = FloatDistribution(low=0, high=0.5, name='delta')
        # Update the delta distribution
        for i, dist in enumerate(hp_space):
            if dist.name == 'delta':
                hp_space[i] = new_delta_distribution
                break

    # Uncomment the next line if you need to set a specific high value for the first hyperparameter
    # TODO: Reset for real run | Set the number of epochs
    # hp_space[0].high = 100

    try:
        # Get the hyperparameters for the current trial
        params = suggest_all(trial, hp_space)

    except Exception as e:
        print(e)


    trial_obj["params"] = params

    # params = {'n_iter': 1, 'lr': 0.0001, 'decoder_n_layers_hidden': 3, 'weight_decay': 0.001, 'batch_size': 256, 'n_units_embedding': 250, 'decoder_n_units_hidden': 250, 'decoder_nonlin': 'tanh',
    #           'decoder_dropout': 0.07905995141252627, 'encoder_n_layers_hidden': 1, 'encoder_n_units_hidden': 350, 'encoder_nonlin': 'tanh', 'encoder_dropout': 0.13587014375548792}

    # else:
    # params["n_iter"] = 5

    # try:
    # Evaluate the current set of hyperparameters
    report = Benchmarks.evaluate(
        [(trial_id, synthsizer_name, params)],
        train_dataset,
        test_dataset,
        repeats=1,
        metrics={"stats": [*objective_metrics]}
        # metrics={"detection": ["detection_mlp"]},  # Uncomment for specific metric
    )
    # except Exception as e:
    #     # Handle invalid set of hyperparameters
    #     print(f"Error ({type(e).__name__}): {e}")
    #     print("Parameters:", params)
    #     breakpoint()
    #     raise optuna.TrialPruned()  # Inform Optuna that this trial should be pruned

    # Calculate the average score across all metrics with the specified direction
    score = report[trial_id].query(f"direction == '{objective_direction_str}'")[
        'mean'].mean()

    trial_obj["score"] = score
    trial_obj["time_sec"] = time.time() - trial_start_time

    print(trial_obj["params"])

    if 'workspace' in trial_obj["params"]:
        del trial_obj["params"]['workspace']  # not required

    # save execution data
    with open(out_json_name, "w") as json_file:
        json.dump(opt_dict, json_file)

    return score


# Full dictionary of metrics is:
# {
#   'sanity': ['data_mismatch', 'common_rows_proportion', 'nearest_syn_neighbor_distance', 'close_values_probability', 'distant_values_probability'],
#   'stats': ['jensenshannon_dist', 'chi_squared_test', 'feature_corr', 'inv_kl_divergence', 'ks_test', 'max_mean_discrepancy', 'wasserstein_dist', 'prdc',
#             'alpha_precision', 'survival_km_distance'],
#   'performance': ['linear_model', 'mlp', 'xgb', 'feat_rank_distance'],
#   'detection': ['detection_xgb', 'detection_mlp', 'detection_gmm', 'detection_linear'],
#   'privacy': ['delta-presence', 'k-anonymization', 'k-map', 'distinct l-diversity', 'identifiability_score', 'DomiasMIA_BNAF', 'DomiasMIA_KDE', 'DomiasMIA_prior']
# }
METRICS_TO_MINIMIZE = ['jensenshannon_dist',
                       'wasserstein_dist', 'max_mean_discrepancy']
METRICS_TO_MAXIMIZE = ['chi_squared_test',
                       'inv_kl_divergence', 'ks_test', 'prdc', 'alpha_precision']
# TODO: Reset for real run
# NUM_TRIALS = 2  # 25  # 50, 75, 100


def run_synthcity_optimizer(
    exp_synthsizer_name: str,
    exp_dataset_name: str,
    exp_train_dataset,
    exp_test_dataset,
    output_path,
    n_trials: int = 25,
    objective_direction: StudyDirection = StudyDirection.MINIMIZE,
) -> None:
    """
    Optimize the given synthsizer based on provided datasets.

    Args:
    - exp_synthsizer_name: Name of the synthsizer.
    - exp_train_dataset: Training dataset.
    - exp_test_dataset: Test dataset.
    - direction: Direction of the study (minimize or maximize).
    - n_trials: Number of trials for optimization.

    Returns:
    None
    """
    # TODO: handle case when
    # 1) no target
    target_col = ML_TASKS_TARGET_CLASS[exp_dataset_name]

    train_loader = GenericDataLoader(
        exp_train_dataset, target_column=target_col)
    test_loader = GenericDataLoader(exp_test_dataset, target_column=target_col)

    # Determine objective metrics based on direction
    if objective_direction.value == StudyDirection.MINIMIZE.value:
        objective_metrics = METRICS_TO_MINIMIZE
    elif objective_direction.value == StudyDirection.MAXIMIZE.value:
        objective_metrics = METRICS_TO_MAXIMIZE
    else:
        objective_metrics = METRICS_TO_MINIMIZE + METRICS_TO_MAXIMIZE

    opt_dict = {}
    opt_dict["synthesizer"] = exp_synthsizer_name
    opt_dict["dataset"] = exp_dataset_name
    opt_dict["objective_metrics"] = objective_metrics
    opt_dict["objective_direction"] = objective_direction.value
    opt_dict["num_trials"] = n_trials
    opt_dict["trials"] = {}

    # save execution data
    out_json_name = f"{output_path}{exp_dataset_name}_{exp_synthsizer_name}_optimiser.json"
    with open(out_json_name, "w") as json_file:
        json.dump(opt_dict, json_file)

    # Create study and optimize
    study = optuna.create_study(direction=objective_direction.value)
    start_time = time.time()
    study.optimize(lambda trial: objective(trial, exp_synthsizer_name,
                   train_loader, test_loader, objective_metrics, objective_direction.value, opt_dict, out_json_name), n_trials=n_trials)
    total_time = time.time() - start_time

    print(f"Set objective metrics: {objective_metrics}")
    print(
        f"Time taken for {exp_synthsizer_name} hyperparameter optimisation with {n_trials} trials: {total_time} seconds")

    opt_dict["total_time_sec"] = total_time

    # TODO: weird delta value is updated; should be in range 0 <= delta <= 0.5
    # if exp_synthsizer_name == "arf":
    #     study.best_params["delta"] = 0.5  # np.random.uniform(0, 0.5)  # 0.5

    opt_dict["best_params"] = study.best_params

    print("Saving opt_dict: ", opt_dict)

    # save execution data
    with open(out_json_name, "w") as json_file:
        json.dump(opt_dict, json_file)

    try:
        return study.best_params
    except Exception as e:
        print(e)
        return None
