from .base_discord_model import BaseDiscordModel

import numpy as np
import scipy.linalg as linalg
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.api import ARIMA


class VarLikeModel(BaseDiscordModel):
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

        # we use ARMA model in case of 1d time series
        if series_dim == 1:
            model_left = ARIMA(left_part, order=(1, 0, 1), trend='c', enforce_stationarity=False)
            model_right = ARIMA(right_part, order=(1, 0, 1), trend='c', enforce_stationarity=False)

            left_result = model_left.fit(
                method='statespace', 
                method_kwargs = {'method': 'lbfgs', 'maxiter': 15}
            )
            right_result = model_right.fit(
                method='statespace', 
                method_kwargs = {'method': 'lbfgs', 'maxiter': 15}
            )

            # compute diffrences between found model paramteres 
            discord_score = linalg.norm(left_result.polynomial_ar - right_result.polynomial_ar)
            discord_score += linalg.norm(left_result.polynomial_ma - right_result.polynomial_ma)
        else:
            model_left = VAR(left_part.T)
            model_right = VAR(right_part.T)

            left_result = model_left.fit(
                maxlags=1
            )

            right_result = model_right.fit(
                maxlags=1
            )

            raise NotImplemented()

        return discord_score