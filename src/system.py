"""
Adaptive Electricity Forecasting System
-------------------------------------

Connects:

Forecasting Engine
        ↓
Actual Observation
        ↓
Error Evaluation
        ↓
Drift Monitoring
        ↓
Persistent Change Detection
        ↓
Adaptive Retraining

Strictly follows the no-lookahead principle:
forecast first → actual arrives → evaluate → monitor → adapt.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from .forecasters import DynamicEnsemble
from .monitor import DriftMonitor
from .adapter import AdaptationController


class AdaptiveRetailSystem:
    """
    Main controller for the adaptive forecasting system.
    """

    def __init__(
        self,
        season_length: int = 7,
        forecast_error_window: int = 30,
        reference_window: int = 30,
        detection_window: int = 10,
        persistence_required: int = 3,
        adaptation_cooldown: int = 10,
        adaptation_window: int = 60,
    ):

        self.forecaster = DynamicEnsemble(
            season_length=season_length,
            error_window=forecast_error_window,
        )

        self.monitor = DriftMonitor(
            reference_window=reference_window,
            detection_window=detection_window,
            persistence_required=persistence_required,
        )

        self.adapter = AdaptationController(
            cooldown_steps=adaptation_cooldown,
            recent_window=adaptation_window,
        )

        self.values: List[float] = []

        self.initialized = False

        self.last_forecast: Optional[float] = None
        self.last_actual: Optional[float] = None
        self.last_error: Optional[float] = None

        self.step_number = 0

    # ========================================================
    # INITIALIZE
    # ========================================================

    def initialize(self, historical_values: List[float]) -> Dict:

        if len(historical_values) < 2:
            raise ValueError(
                "At least two historical observations are required."
            )

        self.values = [
            float(value)
            for value in historical_values
        ]

        # Train forecasting models
        self.forecaster.fit(self.values)

        # IMPORTANT:
        # Give the drift monitor the same historical context.
        # Without this, the monitor would incorrectly start
        # from an empty history after system initialization.
        for value in self.values:
            self.monitor.history.append(value)

        self.initialized = True

        return {
            "initialized": True,
            "data_points": len(self.values),
            "message": "Adaptive forecasting system initialized.",
        }

    # ========================================================
    # FORECAST
    # ========================================================

    def forecast(self) -> Dict:

        if not self.initialized:
            raise RuntimeError(
                "System must be initialized before forecasting."
            )

        result = self.forecaster.forecast(
            self.values
        )

        self.last_forecast = result["forecast"]

        return {
            "step": self.step_number + 1,
            "forecast": result["forecast"],
            "model_predictions": result["model_predictions"],
            "model_weights": result["model_weights"],
            "data_points": len(self.values),
        }

    # ========================================================
    # OBSERVE ACTUAL
    # ========================================================

    def observe(self, actual: float) -> Dict:

        if not self.initialized:
            raise RuntimeError(
                "System must be initialized before observation."
            )

        if self.last_forecast is None:
            raise RuntimeError(
                "A forecast must be generated before observing actual data."
            )

        actual = float(actual)

        forecast = float(self.last_forecast)

        error = abs(
            actual - forecast
        )

        # ----------------------------------------------------
        # UPDATE FORECASTER
        # ----------------------------------------------------

        forecast_update = self.forecaster.observe(
            actual=actual,
            values=self.values,
        )

        # ----------------------------------------------------
        # ADD ACTUAL TO HISTORY
        # ----------------------------------------------------

        self.values.append(actual)

        self.last_actual = actual
        self.last_error = error

        # ----------------------------------------------------
        # DRIFT MONITORING
        # ----------------------------------------------------

        monitor_result = self.monitor.update(
            actual=actual,
            forecast=forecast,
        )

        # ----------------------------------------------------
        # ADAPTATION DECISION
        # ----------------------------------------------------

        adapted = False
        adaptation_result = None

        if self.adapter.should_adapt(
            monitor_result.persistent_change
        ):

            adaptation_result = self.adapter.adapt(
                forecasting_engine=self.forecaster,
                values=self.values,
                drift_score=monitor_result.drift_score,
            )

            adapted = True

        self.adapter.step()

        self.step_number += 1

        # ----------------------------------------------------
        # RESET FORECAST STATE
        # ----------------------------------------------------

        self.last_forecast = None

        return {
            "step": self.step_number,
            "actual": actual,
            "forecast": forecast,
            "error": error,

            "model_errors": forecast_update[
                "model_errors"
            ],

            "updated_weights": forecast_update[
                "updated_weights"
            ],

            "drift": {
                "score": monitor_result.drift_score,
                "change_detected": monitor_result.change_detected,
                "persistent_change": monitor_result.persistent_change,
                "anomaly_score": monitor_result.anomaly_score,
                "mean_shift": monitor_result.mean_shift,
                "volatility_shift": monitor_result.volatility_shift,
                "status": monitor_result.status,
            },

            "adaptation": {
                "triggered": adapted,
                "details": adaptation_result,
            },

            "system_status": (
                "ADAPTING"
                if adapted
                else monitor_result.status
            ),
        }

    # ========================================================
    # COMPLETE STEP
    # ========================================================

    def process_observation(
        self,
        actual: float,
    ) -> Dict:

        # IMPORTANT:
        # Forecast is generated BEFORE actual is revealed.
        forecast_result = self.forecast()

        observation_result = self.observe(
            actual
        )

        return {
            "forecast": forecast_result,
            "observation": observation_result,
        }

    # ========================================================
    # CURRENT STATUS
    # ========================================================

    def get_status(self) -> Dict:

        return {
            "initialized": self.initialized,
            "step": self.step_number,
            "data_points": len(self.values),

            "last_forecast": self.last_forecast,
            "last_actual": self.last_actual,
            "last_error": self.last_error,

            "monitor": self.monitor.get_status(),

            "adaptation": self.adapter.get_status(),

            "models": self.forecaster.get_model_status(),
        }

    # ========================================================
    # HISTORY
    # ========================================================

    def get_history(self) -> List[float]:

        return list(self.values)

    # ========================================================
    # ADAPTATION EVENTS
    # ========================================================

    def get_adaptation_events(self) -> List[Dict]:

        return self.adapter.get_events()