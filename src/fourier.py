from .base_discord_model import BaseDiscordModel

import numpy as np
import scipy.linalg as linalg
from scipy.fft import fft, fftn


class FourierModel(BaseDiscordModel):
    def __init__(self, time_series: np.ndarray):
        super().__init__(time_series)


    def _get_local_discord_score(
            self,
            left_part: np.ndarray, 
            right_part: np.ndarray
    ):
        """

        Args:
            left_part (np.ndarray): time series, shape = (t, num_features)
            right_part (np.ndarray): time series,  shape = (t, num_features)

        Returns:
            (float): discord score for given data
        """
        # dimensionality of series samples
        series_dim = self.time_series.shape[1]

        if series_dim == 1:
            left_fft = fft(left_part)
            right_fft = fft(right_part)
        else:
            left_fft = fftn(left_part)
            right_fft = fftn(right_part)

        discord_score = linalg.norm(left_fft - right_fft) / left_part.shape[0]

        return discord_score