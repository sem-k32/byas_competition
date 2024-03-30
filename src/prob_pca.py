from .base_discord_model import BaseDiscordModel

import numpy as np
import scipy.linalg as linalg
from sklearn.decomposition import FactorAnalysis


class ProbalisitcPcaModel(BaseDiscordModel):
    def __init__(self, time_series: np.ndarray):
        super().__init__(time_series)


    def _get_local_discord_score(
            self,
            left_part: np.ndarray, 
            right_part: np.ndarray
    ):
        """

        Args:
            left_part (np.ndarray): shape = (t, num_features)
            right_part (np.ndarray): shape = (t, num_features)

        Returns:
            (float): discord score for given data
        """
        # dimensionality of series samples
        series_dim = self.time_series.shape[1]

        # create linear models for both series sides
        left_model = FactorAnalysis(series_dim)
        right_model = FactorAnalysis(series_dim)

        left_model.fit(left_part)
        right_model.fit(right_part)

        # count residuals of means and covariance matrices
        mean_residual = left_model.mean_ - right_model.mean_
        covariance_residual = left_model.get_covariance() - right_model.get_covariance()

        # count norm of mean_residual + covariance residual
        discord_score = linalg.norm(mean_residual) + linalg.norm(covariance_residual, ord='fro')

        return discord_score
        

