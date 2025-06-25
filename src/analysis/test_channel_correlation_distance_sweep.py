
from pathlib import Path
import gzip
import pickle

import numpy as np
import matplotlib.pyplot as plt

from src.config.config import Config
from src.data.satellite_manager import SatelliteManager
from src.data.user_manager import UserManager
from src.utils.update_sim import update_sim
from src.utils.calc_channel_correlation import calc_channel_correlation
from src.utils.format_value import format_value


def test_channel_correlation_user_sweep(
        distance_sweep_range: np.ndarray,
        user_1_id: int,
        user_2_id: int,
        disable_wiggle: bool,
        monte_carlo_iterations: int,
) -> None:

    config = Config()
    satellite_manager = SatelliteManager(config)
    user_manager = UserManager(config)
    update_sim(config, satellite_manager, user_manager)

    mean_channel_correlations = np.zeros(len(distance_sweep_range))
    std_channel_correlations = np.zeros(len(distance_sweep_range))

    if disable_wiggle:
        config.user_dist_bound = 0

    for distance_id, distance in enumerate(distance_sweep_range):

        config.user_dist_average = distance
        distance_correlations = np.zeros(monte_carlo_iterations)

        for iteration in range(monte_carlo_iterations):

            update_sim(config, satellite_manager, user_manager)

            distance_correlations[iteration] = abs(calc_channel_correlation(
                channel_1=satellite_manager.channel_state_information[user_1_id, :],
                channel_2=satellite_manager.channel_state_information[user_2_id, :]
            ))

        mean_channel_correlations[distance_id] = np.mean(distance_correlations)
        std_channel_correlations[distance_id] = np.std(distance_correlations)

    # save
    results_path = Path(
        config.output_metrics_path,
        'channel_correlation',
        'user_distance_sweep',
    )
    name = f'{config.sat_nr}sat_{config.sat_tot_ant_nr}ant_{format_value(config.sat_dist_average)}_{format_value(distance_sweep_range[0])}-{format_value(distance_sweep_range[-1])}'
    if not disable_wiggle:
        name += f'_wiggle_{config.user_dist_bound}'
    results_path.mkdir(parents=True, exist_ok=True)
    with gzip.open(Path(results_path, name), 'wb') as file:
        pickle.dump([mean_channel_correlations, std_channel_correlations], file=file)


    # plot
    fig, ax = plt.subplots()

    ax.errorbar(
        distance_sweep_range,
        mean_channel_correlations,
        yerr=std_channel_correlations**2,
    )

    ax.set_xlabel('User Distance [m]')
    ax.set_ylabel('Channel Correlation')

    fig.tight_layout()

    plt.show()


if __name__ == '__main__':

    distance_sweep_range = np.linspace(1, 250_000, 100)
    monte_carlo_iterations = 1000
    user_1_id = 0
    user_2_id = 1
    disable_wiggle = True

    test_channel_correlation_user_sweep(distance_sweep_range, user_1_id, user_2_id, disable_wiggle, monte_carlo_iterations)
