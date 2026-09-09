"""Forecast evaluation and adaptation-performance metrics for the retail forecasting system.

This module is intentionally lightweight and depends only on NumPy and the Python
standard library. It does not introduce a new forecasting model or drift detector.
Instead, it evaluates outputs produced by the existing adaptive demand forecasting
system and the inventory planning layer.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import numpy as np

ArrayLike = Union[Sequence[float], np.ndarray]


class ForecastMetrics:
    """Compute standard forecast quality and adaptation-improvement metrics."""

    def __init__(self, rolling_window: int = 7):
        if rolling_window <= 0:
            raise ValueError("rolling_window must be positive.")
        self.rolling_window = int(rolling_window)

    @staticmethod
    def _as_1d_float_array(values: ArrayLike, name: str = "values") -> np.ndarray:
        """Normalize input into a one-dimensional float NumPy array."""

        array = np.asarray(values, dtype=float).reshape(-1)
        array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0)
        return array

    def _coerce_pair(
        self,
        actual: ArrayLike,
        predictions: ArrayLike,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Validate that actual and prediction arrays are comparable."""

        actual_array = self._as_1d_float_array(actual, "actual")
        prediction_array = self._as_1d_float_array(predictions, "predictions")

        if actual_array.size == 0 and prediction_array.size == 0:
            return actual_array, prediction_array

        if actual_array.size != prediction_array.size:
            raise ValueError(
                "actual and predictions must have the same length; "
                f"received {actual_array.size} and {prediction_array.size}."
            )

        return actual_array, prediction_array

    def mae(self, actual: ArrayLike, predictions: ArrayLike) -> float:
        """Calculate the Mean Absolute Error (MAE)."""

        actual_array, prediction_array = self._coerce_pair(actual, predictions)

        if actual_array.size == 0:
            return 0.0

        return float(np.mean(np.abs(actual_array - prediction_array)))

    def rmse(self, actual: ArrayLike, predictions: ArrayLike) -> float:
        """Calculate the Root Mean Squared Error (RMSE)."""

        actual_array, prediction_array = self._coerce_pair(actual, predictions)

        if actual_array.size == 0:
            return 0.0

        squared_error = np.square(actual_array - prediction_array)
        return float(np.sqrt(np.mean(squared_error)))

    def mape(
        self,
        actual: ArrayLike,
        predictions: ArrayLike,
        zero_tolerance: float = 1e-8,
    ) -> float:
        """Calculate MAPE in percent while guarding against division by zero.

        Values with a near-zero absolute actual are treated as safe floor values to
        prevent runtime errors without distorting the metric excessively.
        """

        actual_array, prediction_array = self._coerce_pair(actual, predictions)

        if actual_array.size == 0:
            return 0.0

        safe_actual = np.where(np.abs(actual_array) > zero_tolerance, actual_array, 1.0)
        percentage_error = np.abs((actual_array - prediction_array) / safe_actual)
        return float(np.mean(percentage_error) * 100.0)

    def bias(self, actual: ArrayLike, predictions: ArrayLike) -> float:
        """Measure forecast bias as mean(prediction - actual).

        Positive values indicate systematic over-forecasting; negative values indicate
        systematic under-forecasting.
        """

        actual_array, prediction_array = self._coerce_pair(actual, predictions)

        if actual_array.size == 0:
            return 0.0

        return float(np.mean(prediction_array - actual_array))

    def rolling_mae(
        self,
        actual: ArrayLike,
        predictions: ArrayLike,
        window: Optional[int] = None,
    ) -> List[float]:
        """Calculate MAE over a rolling window for trailing periods."""

        actual_array, prediction_array = self._coerce_pair(actual, predictions)
        window_size = self.rolling_window if window is None else int(window)

        if window_size <= 0:
            raise ValueError("window must be a positive integer.")

        if actual_array.size == 0:
            return []

        window_size = min(window_size, actual_array.size)
        values: List[float] = []

        for idx in range(window_size - 1, actual_array.size):
            start = idx - window_size + 1
            segment_actual = actual_array[start:idx + 1]
            segment_prediction = prediction_array[start:idx + 1]
            values.append(float(np.mean(np.abs(segment_actual - segment_prediction))))

        return values

    def rolling_rmse(
        self,
        actual: ArrayLike,
        predictions: ArrayLike,
        window: Optional[int] = None,
    ) -> List[float]:
        """Calculate RMSE over a rolling window for trailing periods."""

        actual_array, prediction_array = self._coerce_pair(actual, predictions)
        window_size = self.rolling_window if window is None else int(window)

        if window_size <= 0:
            raise ValueError("window must be a positive integer.")

        if actual_array.size == 0:
            return []

        window_size = min(window_size, actual_array.size)
        values: List[float] = []

        for idx in range(window_size - 1, actual_array.size):
            start = idx - window_size + 1
            segment_actual = actual_array[start:idx + 1]
            segment_prediction = prediction_array[start:idx + 1]
            rmse = float(np.sqrt(np.mean(np.square(segment_actual - segment_prediction))))
            values.append(rmse)

        return values

    def compare_models(
        self,
        actual: ArrayLike,
        model_predictions: Mapping[str, ArrayLike],
    ) -> Dict[str, Dict[str, float]]:
        """Compare multiple model predictions using standard forecasting metrics."""

        if not isinstance(model_predictions, Mapping):
            raise TypeError("model_predictions must be a mapping of model names to prediction arrays.")

        comparison: Dict[str, Dict[str, float]] = {}

        for model_name, predictions in model_predictions.items():
            if predictions is None:
                raise ValueError(f"Predictions for model '{model_name}' are missing.")
            comparison[model_name] = {
                "mae": self.mae(actual, predictions),
                "rmse": self.rmse(actual, predictions),
                "mape": self.mape(actual, predictions),
                "bias": self.bias(actual, predictions),
            }

        return comparison

    def adaptation_improvement(
        self,
        before_error: float,
        after_error: float,
    ) -> float:
        """Calculate percentage improvement from before to after an adaptation event.

        Positive values indicate improved performance after adaptation.
        """

        before_value = float(before_error)
        after_value = float(after_error)

        if np.isnan(before_value) or np.isnan(after_value):
            return 0.0

        if abs(before_value) < 1e-12:
            return 0.0 if abs(after_value) < 1e-12 else float("-inf")

        improvement = ((before_value - after_value) / abs(before_value)) * 100.0
        return float(np.nan_to_num(improvement, nan=0.0, posinf=0.0, neginf=0.0))

    def before_after_adaptation_analysis(
        self,
        actual: ArrayLike,
        before_predictions: ArrayLike,
        after_predictions: ArrayLike,
    ) -> Dict[str, Any]:
        """Compare performance before and after an adaptation event."""

        actual_array, before_array = self._coerce_pair(actual, before_predictions)
        _, after_array = self._coerce_pair(actual_array, after_predictions)

        before_metrics = {
            "mae": self.mae(actual_array, before_array),
            "rmse": self.rmse(actual_array, before_array),
            "mape": self.mape(actual_array, before_array),
            "bias": self.bias(actual_array, before_array),
        }

        after_metrics = {
            "mae": self.mae(actual_array, after_array),
            "rmse": self.rmse(actual_array, after_array),
            "mape": self.mape(actual_array, after_array),
            "bias": self.bias(actual_array, after_array),
        }

        improvement = {
            "mae_pct": self.adaptation_improvement(before_metrics["mae"], after_metrics["mae"]),
            "rmse_pct": self.adaptation_improvement(before_metrics["rmse"], after_metrics["rmse"]),
            "mape_pct": self.adaptation_improvement(before_metrics["mape"], after_metrics["mape"]),
        }

        return {
            "before": before_metrics,
            "after": after_metrics,
            "improvement_pct": improvement,
            "improved": (
                after_metrics["mae"] <= before_metrics["mae"]
                and after_metrics["rmse"] <= before_metrics["rmse"]
            ),
        }

    def performance_summary(
        self,
        actual: ArrayLike,
        predictions: ArrayLike,
        rolling_window: Optional[int] = None,
    ) -> Dict[str, float]:
        """Return a compact dictionary of key forecast metrics for dashboards and reports."""

        actual_array, prediction_array = self._coerce_pair(actual, predictions)
        window_size = self.rolling_window if rolling_window is None else int(rolling_window)

        summary = {
            "n_points": int(actual_array.size),
            "mae": self.mae(actual_array, prediction_array),
            "rmse": self.rmse(actual_array, prediction_array),
            "mape": self.mape(actual_array, prediction_array),
            "bias": self.bias(actual_array, prediction_array),
            "mean_actual": float(np.mean(actual_array)) if actual_array.size else 0.0,
            "mean_predicted": float(np.mean(prediction_array)) if prediction_array.size else 0.0,
        }

        if actual_array.size:
            summary["rolling_mae"] = (
                self.rolling_mae(actual_array, prediction_array, window=window_size)[-1]
                if self.rolling_mae(actual_array, prediction_array, window=window_size)
                else 0.0
            )
            summary["rolling_rmse"] = (
                self.rolling_rmse(actual_array, prediction_array, window=window_size)[-1]
                if self.rolling_rmse(actual_array, prediction_array, window=window_size)
                else 0.0
            )
        else:
            summary["rolling_mae"] = 0.0
            summary["rolling_rmse"] = 0.0

        return summary


def mae(actual: ArrayLike, predictions: ArrayLike) -> float:
    """Convenience function for MAE calculations."""

    return ForecastMetrics().mae(actual, predictions)


def rmse(actual: ArrayLike, predictions: ArrayLike) -> float:
    """Convenience function for RMSE calculations."""

    return ForecastMetrics().rmse(actual, predictions)


def mape(actual: ArrayLike, predictions: ArrayLike, zero_tolerance: float = 1e-8) -> float:
    """Convenience function for MAPE calculations with safe zero handling."""

    return ForecastMetrics().mape(actual, predictions, zero_tolerance=zero_tolerance)


def forecast_bias(actual: ArrayLike, predictions: ArrayLike) -> float:
    """Convenience function for signed forecast bias."""

    return ForecastMetrics().bias(actual, predictions)


def rolling_mae(
    actual: ArrayLike,
    predictions: ArrayLike,
    window: int = 7,
) -> List[float]:
    """Convenience function for rolling MAE."""

    return ForecastMetrics(rolling_window=window).rolling_mae(actual, predictions, window=window)


def rolling_rmse(
    actual: ArrayLike,
    predictions: ArrayLike,
    window: int = 7,
) -> List[float]:
    """Convenience function for rolling RMSE."""

    return ForecastMetrics(rolling_window=window).rolling_rmse(actual, predictions, window=window)


def compare_models(
    actual: ArrayLike,
    model_predictions: Mapping[str, ArrayLike],
) -> Dict[str, Dict[str, float]]:
    """Convenience function to compare multiple model predictions."""

    return ForecastMetrics().compare_models(actual, model_predictions)


def adaptation_improvement(before_error: float, after_error: float) -> float:
    """Convenience function to calculate percentage improvement after adaptation."""

    return ForecastMetrics().adaptation_improvement(before_error, after_error)


def before_after_adaptation_analysis(
    actual: ArrayLike,
    before_predictions: ArrayLike,
    after_predictions: ArrayLike,
) -> Dict[str, Any]:
    """Convenience function for before/after adaptation analysis."""

    return ForecastMetrics().before_after_adaptation_analysis(
        actual,
        before_predictions,
        after_predictions,
    )


def performance_summary(
    actual: ArrayLike,
    predictions: ArrayLike,
    rolling_window: Optional[int] = None,
) -> Dict[str, float]:
    """Convenience function returning the core forecast summary dictionary."""

    return ForecastMetrics(rolling_window=rolling_window or 7).performance_summary(
        actual,
        predictions,
        rolling_window=rolling_window,
    )


__all__ = [
    "ForecastMetrics",
    "mae",
    "rmse",
    "mape",
    "forecast_bias",
    "rolling_mae",
    "rolling_rmse",
    "compare_models",
    "adaptation_improvement",
    "before_after_adaptation_analysis",
    "performance_summary",
]
