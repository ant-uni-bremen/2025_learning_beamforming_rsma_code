
import numpy as np


def calc_tx_power_distribution(
    w_precoder: np.ndarray,
) -> np.ndarray:
    """TODO: Comment"""

    w_common = w_precoder[:,0]
    w_private = w_precoder[:,1:]

    user_nr = w_private.shape[1]

    power_per_user_comon_part = 1/user_nr * (np.linalg.norm(w_common))**2

    power_per_user_private_part = np.zeros(user_nr)

    for user_id in range(user_nr):

        norm = np.linalg.norm(w_private[:, user_id])

        power_per_user_private_part[user_id] = norm**2

    power_per_user = power_per_user_comon_part + power_per_user_private_part

    return power_per_user
