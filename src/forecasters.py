"""
Adaptive Retail Demand Forecasting
-----------------------------------

Forecasting engine containing multiple forecasting strategies:

1. Naive Forecast
2. Seasonal Naive Forecast
3. Exponentially Weighted Moving Average (EWMA)
4. Online Linear Regression
5. Dynamic Ensemble

The ensemble continuously evaluates recent model performance
and gives higher weight to models that are currently performing better.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from sklearn.linear_model import SGDRegressor


# ============================================================
# BASE FORECASTER
# ============================================================

class BaseForecaster:
    """Base interface for all forecasting models."""

    name = "Base"

    def fit(self, values: List[float]) -> None:
        raise NotImplementedError

    def predict(self, values: List[float]) -> float:
        raise NotImplementedError

    def update(self, actual: float) -> None:
        """Optional online update."""
        pass


# ============================================================
# 1. NAIVE FORECASTER
# ============================================================

class NaiveForecaster(BaseForecaster):
    """
    Predicts the next value as the most recently observed value.
    """

    name = "Naive"

    def __init__(self):
        self.last_value: Optional[float] = None

    def fit(self, values: List[float]) -> None:
        if values:
            self.last_value = float(values[-1])

    def predict(self, values: List[float]) -> float:
        if not values:
            return 0.0

        self.last_value = float(values[-1])
        return self.last_value


# ============================================================
# 2. SEASONAL NAIVE FORECASTER
# ============================================================

class SeasonalNaiveForecaster(BaseForecaster):
    """
    Predicts using the value observed one complete seasonal cycle ago.

    Example:
    If season_length = 7, Monday's demand is initially predicted
    using the previous Monday's demand.
    """

    name = "Seasonal Naive"

    def __init__(self, season_length: int = 7):
        if season_length < 1:
            raise ValueError("season_length must be >= 1")

        self.season_length = season_length

    def fit(self, values: List[float]) -> None:
        pass

    def predict(self, values: List[float]) -> float:
        if not values:
            return 0.0

        if len(values) < self.season_length:
            return float(values[-1])

        return float(values[-self.season_length])


# ============================================================
# 3. EWMA FORECASTER
# ============================================================

class EWMAForecaster(BaseForecaster):
    """
    Exponentially Weighted Moving Average.

    Recent observations receive more importance than older observations.
    """

    name = "EWMA"

    def __init__(self, alpha: float = 0.3):
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be between 0 and 1")

        self.alpha = alpha
        self.level: Optional[float] = None

    def fit(self, values: List[float]) -> None:
        if not values:
            self.level = None
            return

        level = float(values[0])

        for value in values[1:]:
            level = self.alpha * float(value) + (1 - self.alpha) * level

        self.level = level

    def predict(self, values: List[float]) -> float:
        if not values:
            return 0.0

        self.fit(values)
        return float(self.level)


# ============================================================
# 4. ONLINE LINEAR FORECASTER
# ============================================================

class OnlineLinearForecaster(BaseForecaster):
    """
    Online linear regression using lagged observations.

    The model learns relationships between recent demand values
    and the next demand value.
    """

    name = "Online Linear"

    def __init__(
        self,
        lags: int = 7,
        learning_rate: str = "adaptive",
    ):
        self.lags = lags

        self.model = SGDRegressor(
            loss="huber",
            penalty="l2",
            alpha=0.0001,
            learning_rate=learning_rate,
            eta0=0.01,
            max_iter=1,
            warm_start=True,
            random_state=42,
        )

        self.is_fitted = False

    def _create_training_data(self, values: List[float]):
        values = np.asarray(values, dtype=float)

        if len(values) <= self.lags:
            return None, None

        X = []
        y = []

        for i in range(self.lags, len(values)):
            X.append(values[i - self.lags:i])
            y.append(values[i])

        return np.asarray(X), np.asarray(y)

    def fit(self, values: List[float]) -> None:
        X, y = self._create_training_data(values)

        if X is None:
            self.is_fitted = False
            return

        self.model.fit(X, y)
        self.is_fitted = True

    def predict(self, values: List[float]) -> float:
        if len(values) < self.lags:
            return float(values[-1]) if values else 0.0

        if not self.is_fitted:
            self.fit(values)

        recent = np.asarray(values[-self.lags:], dtype=float).reshape(1, -1)

        prediction = self.model.predict(recent)[0]

        return float(prediction)

    def update(self, actual: float, values: List[float]) -> None:
        """
        Perform an online update after the actual value becomes available.
        """

        if len(values) < self.lags:
            return

        features = np.asarray(
            values[-self.lags:],
            dtype=float
        ).reshape(1, -1)

        target = np.asarray([float(actual)])

        self.model.partial_fit(features, target)
        self.is_fitted = True


# ============================================================
# MODEL PERFORMANCE RECORD
# ============================================================

@dataclass
class ModelPerformance:
    name: str
    recent_error: float = np.inf
    weight: float = 0.25
    predictions: int = 0


# ============================================================
# DYNAMIC ENSEMBLE
# ============================================================

class DynamicEnsemble:
    """
    Combines multiple forecasting models.

    Model weights are dynamically updated according to recent
    forecasting performance.

    Better recent performance → higher weight.
    Worse recent performance → lower weight.
    """

    def __init__(
        self,
        season_length: int = 7,
        error_window: int = 30,
    ):

        self.error_window = error_window

        self.models: Dict[str, BaseForecaster] = {
            "Naive": NaiveForecaster(),
            "Seasonal Naive": SeasonalNaiveForecaster(
                season_length=season_length
            ),
            "EWMA": EWMAForecaster(alpha=0.3),
            "Online Linear": OnlineLinearForecaster(
                lags=season_length
            ),
        }

        self.performance: Dict[str, ModelPerformance] = {
            name: ModelPerformance(name=name)
            for name in self.models
        }

        self.error_history: Dict[str, deque] = {
            name: deque(maxlen=error_window)
            for name in self.models
        }

        self.last_predictions: Dict[str, float] = {}

        self.history: List[float] = []

    # --------------------------------------------------------
    # FIT ALL MODELS
    # --------------------------------------------------------

    def fit(self, values: List[float]) -> None:
        self.history = list(map(float, values))

        for model in self.models.values():
            model.fit(self.history)

    # --------------------------------------------------------
    # GENERATE INDIVIDUAL PREDICTIONS
    # --------------------------------------------------------

    def predict_all(self, values: List[float]) -> Dict[str, float]:

        predictions = {}

        for name, model in self.models.items():

            try:
                prediction = model.predict(values)

                # Prevent invalid predictions
                if not np.isfinite(prediction):
                    prediction = float(values[-1])

            except Exception:
                prediction = float(values[-1]) if values else 0.0

            predictions[name] = float(prediction)

        self.last_predictions = predictions

        return predictions

    # --------------------------------------------------------
    # CALCULATE DYNAMIC WEIGHTS
    # --------------------------------------------------------

    def _calculate_weights(self) -> Dict[str, float]:

        scores = {}

        for name in self.models:

            errors = self.error_history[name]

            if not errors:
                scores[name] = 1.0
                continue

            recent_error = float(np.mean(errors))

            # Small error → large score
            scores[name] = 1.0 / (recent_error + 1e-6)

        total = sum(scores.values())

        if total <= 0:
            equal_weight = 1.0 / len(scores)

            return {
                name: equal_weight
                for name in scores
            }

        weights = {
            name: score / total
            for name, score in scores.items()
        }

        return weights

    # --------------------------------------------------------
    # ENSEMBLE FORECAST
    # --------------------------------------------------------

    def forecast(self, values: List[float]):

        predictions = self.predict_all(values)

        weights = self._calculate_weights()

        ensemble_prediction = sum(
            predictions[name] * weights[name]
            for name in predictions
        )

        # Retail demand cannot normally be negative
        ensemble_prediction = max(0.0, ensemble_prediction)

        for name in self.models:

            self.performance[name].weight = weights[name]

        return {
            "forecast": float(ensemble_prediction),
            "model_predictions": predictions,
            "model_weights": weights,
        }

    # --------------------------------------------------------
    # OBSERVE ACTUAL VALUE
    # --------------------------------------------------------

    def observe(
        self,
        actual: float,
        values: List[float],
    ) -> Dict:

        actual = float(actual)

        errors = {}

        for name, prediction in self.last_predictions.items():

            error = abs(prediction - actual)

            errors[name] = float(error)

            self.error_history[name].append(error)

            self.performance[name].recent_error = float(
                np.mean(self.error_history[name])
            )

            self.performance[name].predictions += 1

        # Online update after actual value becomes available
        online_model = self.models.get("Online Linear")

        if isinstance(online_model, OnlineLinearForecaster):

            online_model.update(
                actual=actual,
                values=values,
            )

        # Update history
        self.history.append(actual)

        return {
            "actual": actual,
            "model_errors": errors,
            "updated_weights": self._calculate_weights(),
        }

    # --------------------------------------------------------
    # CURRENT MODEL STATUS
    # --------------------------------------------------------

    def get_model_status(self) -> Dict:

        return {
            name: {
                "recent_error": performance.recent_error,
                "weight": performance.weight,
                "predictions": performance.predictions,
            }
            for name, performance in self.performance.items()
        }

    # --------------------------------------------------------
    # RESET / ADAPTATION SUPPORT
    # --------------------------------------------------------

    def reset_weights(self) -> None:

        equal_weight = 1.0 / len(self.models)

        for performance in self.performance.values():
            performance.weight = equal_weight

    def retrain_recent_window(
        self,
        values: List[float],
        window_size: int = 60,
    ) -> None:

        recent_values = list(values[-window_size:])

        if len(recent_values) < 2:
            return

        self.history = recent_values

        for model in self.models.values():

            try:
                model.fit(recent_values)
            except Exception:
                pass

        self.reset_weights()

        self.error_history = {
            name: deque(maxlen=self.error_window)
            for name in self.models
        }

        self.last_predictions = {}