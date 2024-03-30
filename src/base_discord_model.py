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

    def compute_discord_scores(
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
    

    def find_discordance_point(
        self,
        window_size: int
    ):
        """
            Dichotomy method based on mean diffrences of discord scores between series parts
        """
        discord_scores = self.compute_discord_scores(window_size)

        left_bound = 0
        right_bound = len(discord_scores) - 1

        series_len = self.time_series.shape[0]

        while right_bound - left_bound > 4:
            lefter_point = left_bound + (right_bound - left_bound) // 4
            righter_point = right_bound - (right_bound - left_bound) // 4

            mean_diff_1 = np.mean(discord_scores[left_bound:lefter_point]) - np.mean(discord_scores[lefter_point:right_bound])
            mean_diff_1 = np.abs(mean_diff_1)
            mean_diff_2 = np.mean(discord_scores[left_bound:righter_point]) - np.mean(discord_scores[righter_point:right_bound])
            mean_diff_2 = np.abs(mean_diff_2)

            if mean_diff_1 > mean_diff_2:
                right_bound = righter_point
            else:
                left_bound = lefter_point

        final_diffrence = np.abs(np.mean(discord_scores[:left_bound]) - np.mean(discord_scores[right_bound:]))

        if final_diffrence > 0.4:
            discord_point = left_bound + (right_bound - left_bound) // 2
        else:
            discord_point = series_len + 1

        # compute final point in time
        if discord_point == series_len + 1:
            discord_point_ts = series_len + 1
        else:
            discord_point_ts = (window_size - 1) + discord_point  * window_size

        # return point and level of confidence
        return discord_point_ts, final_diffrence


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