"""
Drift and Change Monitoring
---------------------------

Detects meaningful changes in retail demand while avoiding
unnecessary adaptation to isolated anomalies.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Dict, List

import numpy as np


@dataclass
class MonitorResult:
    drift_score: float
    change_detected: bool
    persistent_change: bool
    anomaly_score: float
    mean_shift: float
    volatility_shift: float
    status: str


class DriftMonitor:
    """
    Multi-signal drift detector.

    The monitor compares recent demand behaviour with an older
    reference window.

    Signals:
    1. Mean shift
    2. Volatility shift
    3. Recent forecast error
    4. Persistence of the change

    A single abnormal observation should not immediately trigger
    adaptation.
    """

    def __init__(
        self,
        reference_window: int = 30,
        detection_window: int = 10,
        persistence_required: int = 3,
    ):

        self.reference_window = reference_window
        self.detection_window = detection_window
        self.persistence_required = persistence_required

        self.history = deque(
            maxlen=reference_window + detection_window
        )

        self.drift_history = deque(maxlen=20)

        self.consecutive_drift = 0

    # --------------------------------------------------------
    # MAIN UPDATE
    # --------------------------------------------------------

    def update(
        self,
        actual: float,
        forecast: float,
    ) -> MonitorResult:

        actual = float(actual)
        forecast = float(forecast)

        self.history.append(actual)

        values = list(self.history)

        if len(values) < self.reference_window + self.detection_window:

            result = MonitorResult(
                drift_score=0.0,
                change_detected=False,
                persistent_change=False,
                anomaly_score=0.0,
                mean_shift=0.0,
                volatility_shift=0.0,
                status="WARMING UP",
            )

            self.drift_history.append(0.0)

            return result

        reference = np.asarray(
            values[
                : -self.detection_window
            ],
            dtype=float,
        )

        recent = np.asarray(
            values[
                -self.detection_window:
            ],
            dtype=float,
        )

        # ----------------------------------------------------
        # 1. MEAN SHIFT
        # ----------------------------------------------------

        reference_mean = np.mean(reference)
        recent_mean = np.mean(recent)

        reference_std = np.std(reference) + 1e-6

        mean_shift = abs(
            recent_mean - reference_mean
        ) / reference_std

        # ----------------------------------------------------
        # 2. VOLATILITY SHIFT
        # ----------------------------------------------------

        reference_volatility = np.std(reference) + 1e-6
        recent_volatility = np.std(recent)

        volatility_shift = abs(
            recent_volatility - reference_volatility
        ) / reference_volatility

        # ----------------------------------------------------
        # 3. FORECAST ERROR
        # ----------------------------------------------------

        forecast_error = abs(
            actual - forecast
        )

        anomaly_score = forecast_error / (
            reference_std + 1e-6
        )

        # ----------------------------------------------------
        # 4. COMBINED DRIFT SCORE
        # ----------------------------------------------------

        drift_score = (
            0.45 * min(mean_shift / 3.0, 1.0)
            + 0.25 * min(volatility_shift / 2.0, 1.0)
            + 0.30 * min(anomaly_score / 3.0, 1.0)
        )

        drift_score = float(
            np.clip(drift_score, 0.0, 1.0)
        )

        self.drift_history.append(drift_score)

        # ----------------------------------------------------
        # CHANGE DETECTION
        # ----------------------------------------------------

        change_detected = drift_score >= 0.55

        if change_detected:
            self.consecutive_drift += 1
        else:
            self.consecutive_drift = 0

        persistent_change = (
            self.consecutive_drift
            >= self.persistence_required
        )

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        if persistent_change:
            status = "PERSISTENT CHANGE"

        elif change_detected:
            status = "POSSIBLE CHANGE"

        elif drift_score >= 0.30:
            status = "WATCH"

        else:
            status = "STABLE"

        return MonitorResult(
            drift_score=drift_score,
            change_detected=change_detected,
            persistent_change=persistent_change,
            anomaly_score=float(anomaly_score),
            mean_shift=float(mean_shift),
            volatility_shift=float(volatility_shift),
            status=status,
        )

    # --------------------------------------------------------
    # RESET
    # --------------------------------------------------------

    def reset(self) -> None:

        self.history.clear()
        self.drift_history.clear()
        self.consecutive_drift = 0

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def get_status(self) -> Dict:

        return {
            "drift_score": (
                self.drift_history[-1]
                if self.drift_history
                else 0.0
            ),
            "consecutive_drift": self.consecutive_drift,
            "history_size": len(self.history),
        }