
import numpy as np


def calc_channel_correlation(
        channel_1: np.ndarray,
        channel_2: np.ndarray,
        normalize: bool = True,
) -> np.ndarray:

    correlation = channel_1 @ channel_2[np.newaxis].conj().T

    if normalize:

        correlation = correlation / (
            np.linalg.norm(channel_1)
            *
            np.linalg.norm(channel_2)
        )

    return correlation[0]
