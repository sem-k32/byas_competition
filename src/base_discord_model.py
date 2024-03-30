from abc import ABC, abstractmethod
import numpy as np


class BaseDiscordModel(ABC):
    def __init__(self, time_series: np.ndarray):
        self.time_series = time_series

    @abstractmethod
    def _get_local_discord_score(
            self,
            left_part: np.ndarray, 
            right_part: np.ndarray
    ):
        ...

    def find_discord(
            self,
            window_size: int, 
    ) -> np.ndarray:
        """compute discord function over time series
           window stride = window_size

        Args:
            window_size (int): the discord will be computed within its vicinity

        Returns:
            np.ndarray: discord scores
        """
        # dimensionality of series samples
        series_dim = self.time_series.shape[1]

        series_len = self.time_series.shape[0]

        # container for discord scores along the series
        discord_scores = []

        window_center = window_size - 1

        while window_center + (window_size - 1) < series_len:
            cur_left = self.time_series[window_center - window_size + 1: window_center + 1]
            cur_right = self.time_series[window_center : window_center + window_size]

            discord_scores.append(self._get_local_discord_score(
                cur_left,
                cur_right
            ))

            # compute position of new window center
            window_center += window_size

        return discord_scores


"""
def find_discord(
            self,
            window_size: int, 
    ) -> np.ndarray:
        # dimensionality of series samples
        series_dim = self.time_series.shape[1]

        series_len = self.time_series.shape[0]

        # container for discord scores along the series
        discord_scores = np.empty(series_len - (2* window_size - 1) + 1)

        for i, window_center in enumerate(range(window_size - 1, series_len - window_size + 1)):
            cur_left = self.time_series[window_center - window_size + 1: window_center + 1]
            cur_right = self.time_series[window_center : window_center + window_size]

            discord_scores[i] = self._get_local_discord_score(
                cur_left,
                cur_right
            )

        return discord_scores
"""