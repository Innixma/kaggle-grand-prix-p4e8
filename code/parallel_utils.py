import os

import numpy as np


# --- Parameters ---
"""The Ray address to connect to. Provide "auto" to start a new cluster."""
ray_address: str | None = None

"""If True, we train all folds (and repeated folds) of a model for AutoGluon at the same time."""
autogluon_force_full_repeated_cross_validation: bool = True

"""If AutoGluon is run in a distributed mode across several nodes.
If True, all relevant parameters are set correctly. If False, not."""
autogluon_distributed: bool = True

"""If AutoGluon is runs several different model fits in parallel. This will result in fitting all folds of a
all `autogluon_distributed_ray_worker` many models at the same time."""
autogluon_distributed_parallel_model_fit: str = "True"

"""If AutoGluon is runs several different model predictions in parallel. This results in predicting all models from
all bags in parallel with `autogluon_distributed_ray_worker` many bagged models at the same time."""
autogluon_distributed_parallel_model_predict: str = "True"

autogluon_distributed_ray_worker_predict: int = 12

num_cpus_per_instance = 192
num_instances = 1
num_bag_folds = 8
num_bag_sets = 1
num_cpus_per_fold = 8

total_cpus = num_cpus_per_instance * num_instances

num_parallel_bags = int(np.floor(total_cpus/(num_bag_folds*num_bag_sets*num_cpus_per_fold)))


def setup_parallel(parallel_fit: bool = False):
    ag_args_fit = {}
    ag_args_ensemble = {}
    # ag_args_ensemble = {"max_models_per_type": 3}

    if parallel_fit:
        num_cpus_per_fold = int(np.floor(total_cpus / num_parallel_bags / (num_bag_folds * num_bag_sets)))

        print(f"{total_cpus}\t= total_cpus")
        print(f"{num_cpus_per_fold}\t= num_cpus_per_fold")
        print(f"Num parallel bags: {num_parallel_bags}")

        ag_args_fit["num_cpus"] = num_cpus_per_fold

        os.environ["AG_DISTRIBUTED_N_RAY_WORKERS"] = f"{num_parallel_bags}"
        os.environ["AG_DISTRIBUTED_FIT_MODELS_PARALLEL"] = autogluon_distributed_parallel_model_fit
        os.environ["AG_DISTRIBUTED_N_RAY_WORKERS_PREDICT"] = str(autogluon_distributed_ray_worker_predict)
        os.environ["AG_DISTRIBUTED_PREDICT_MODELS_PARALLEL"] = autogluon_distributed_parallel_model_predict

    return ag_args_fit, ag_args_ensemble
