"test scirpt to evaluate learned performances"

import numpy as np


from src.config.config import Config
from pathlib import Path
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_error_sweep
from src.analysis.helpers.test_learned_precoder import test_sac_precoder_error_sweep, \
    test_sac_precoder_user_distance_sweep
from src.analysis.helpers.test_learned_precoder import test_learned_rsma_complete_user_distance_sweep
from src.analysis.helpers.test_learned_precoder import test_learned_rsma_power_common_user_distance_sweep
from src.analysis.helpers.test_mmse_precoder import test_mmse_precoder_user_distance_sweep
from src.analysis.helpers.test_rsma_precoder import test_rsma_precoder_user_distance_sweep
from src.analysis.helpers.test_mrc_precoder import test_mrc_precoder_user_distance_sweep

error_sweep_range = np.arange(0,0.11, 0.01)
user_dist_sweep_range = np.arange(500,50500, 500)

monte_carlo_iterations = 5000
config = Config()

# test_mmse_precoder_user_distance_sweep(
#     config=config,
#     distance_sweep_range=user_dist_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics=['sumrate', 'fairness'],
# )

# test_rsma_precoder_user_distance_sweep(
#     config=config,
#     distance_sweep_range=user_dist_sweep_range,
#     rsma_factor=0.75,
#     common_part_precoding_style='basic',
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics=['sumrate','fairness'],
#     )


# model_path= Path(config.trained_models_path, 'RSMA_Journal', '01_sum_rate','power_common_rsma_snap_2.917' )
#
# test_learned_rsma_power_common_user_distance_sweep(
#     config=config,
#     model_path=model_path,
#     distance_sweep_range=user_dist_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics=['sumrate', 'fairness'],)


# model_path= Path(config.trained_models_path, 'RSMA_Journal', '01_mixed_objective_error','full_snap_3.764' )
#
# test_sac_precoder_user_distance_sweep(
#     config=config,
#     model_path=model_path,
#     distance_sweep_range=user_dist_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics=['sumrate','fairness'],
# )

model_path= Path(config.trained_models_path, 'RSMA_Journal', '01_mixed_objective_error', 'full_rsma_snap_3.863')

test_learned_rsma_complete_user_distance_sweep(
    config=config,
    model_path=model_path,
    distance_sweep_range=user_dist_sweep_range,
    monte_carlo_iterations=monte_carlo_iterations,
    metrics=['sumrate','fairness'],
)

# model_path= Path(config.trained_models_path, 'RSMA_Journal', 'full_rsma_snap_2.871')
#
# test_learned_rsma_complete_user_distance_sweep(
#     config=config,
#     model_path=model_path,
#     distance_sweep_range=user_dist_sweep_range,
#     monte_carlo_iterations=monte_carlo_iterations,
#     metrics='sumrate',
# )