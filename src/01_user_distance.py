"test scirpt to evaluate learned performances"

import numpy as np


from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_error_sweep, \
    test_sac_precoder_user_distance_sweep
from src.analysis.helpers.test_learned_precoder import test_learned_rsma_complete_user_distance_sweep
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_user_distance_sweep
from src.analysis.helpers.test_mrc_precoder import test_mrc_precoder_user_distance_sweep

error_sweep_range = np.arange(0,0.11, 0.01)
user_dist_sweep_range = np.arange(0,51000, 1000)
monte_carlo_iterations = 1000
config = Config()

# test_mmse_precoder_user_distance_sweep(
#     config=config,
#     distance_sweep_range=user_dist_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     # metrics=metrics
# )

#
#
#
# model_path= Path(config.trained_models_path, 'RSMA_Journal', 'full_snap_4.053')
#
# test_sac_precoder_user_distance_sweep(
#     config=config,
#     model_path=model_path,
#     distance_sweep_range=user_dist_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
# )

model_path= Path(config.trained_models_path, 'RSMA_Journal', 'full_rsma_snap_4.018')

test_learned_rsma_complete_user_distance_sweep(
    config=config,
    model_path=model_path,
    distance_sweep_range=user_dist_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics='fairness',
)

model_path= Path(config.trained_models_path, 'RSMA_Journal', 'full_rsma_snap_2.865')

test_learned_rsma_complete_user_distance_sweep(
    config=config,
    model_path=model_path,
    distance_sweep_range=user_dist_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics='fairness',
)