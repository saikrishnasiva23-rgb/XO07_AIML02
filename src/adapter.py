"""
Adaptive Model Controller
-------------------------

Controls when the forecasting system should adapt
after meaningful and persistent demand changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List


@dataclass
class AdaptationEvent:
    timestamp: str
    reason: str
    drift_score: float
    action: str
    data_points_used: int


class AdaptationController:
    """
    Decides when the forecasting engine should adapt.

    Adaptation occurs only when persistent change is confirmed.
    """

    def __init__(
        self,
        cooldown_steps: int = 10,
        recent_window: int = 60,
    ):

        self.cooldown_steps = cooldown_steps
        self.recent_window = recent_window

        self.steps_since_adaptation = cooldown_steps

        self.events: List[AdaptationEvent] = []

    # --------------------------------------------------------
    # SHOULD ADAPT?
    # --------------------------------------------------------

    def should_adapt(
        self,
        persistent_change: bool,
    ) -> bool:

        if not persistent_change:
            return False

        if self.steps_since_adaptation < self.cooldown_steps:
            return False

        return True

    # --------------------------------------------------------
    # PERFORM ADAPTATION
    # --------------------------------------------------------

    def adapt(
        self,
        forecasting_engine,
        values: List[float],
        drift_score: float,
        reason: str = "Persistent demand change detected",
    ) -> Dict:

        recent_values = list(values[-self.recent_window:])

        forecasting_engine.retrain_recent_window(
            recent_values,
            window_size=self.recent_window,
        )

        self.steps_since_adaptation = 0

        event = AdaptationEvent(
            timestamp=datetime.now().isoformat(
                timespec="seconds"
            ),
            reason=reason,
            drift_score=float(drift_score),
            action="Recent-window retraining + dynamic reweighting",
            data_points_used=len(recent_values),
        )

        self.events.append(event)

        return {
            "adapted": True,
            "reason": reason,
            "drift_score": float(drift_score),
            "action": event.action,
            "data_points_used": len(recent_values),
            "timestamp": event.timestamp,
        }

    # --------------------------------------------------------
    # STEP COUNTER
    # --------------------------------------------------------

    def step(self) -> None:
        self.steps_since_adaptation += 1

    # --------------------------------------------------------
    # EVENT HISTORY
    # --------------------------------------------------------

    def get_events(self) -> List[Dict]:

        return [
            {
                "timestamp": event.timestamp,
                "reason": event.reason,
                "drift_score": event.drift_score,
                "action": event.action,
                "data_points_used": event.data_points_used,
            }
            for event in self.events
        ]

    # --------------------------------------------------------
    # STATUS
    # --------------------------------------------------------

    def get_status(self) -> Dict:

        return {
            "adaptation_count": len(self.events),
            "steps_since_adaptation": self.steps_since_adaptation,
            "cooldown_steps": self.cooldown_steps,
            "last_adaptation": (
                self.events[-1].timestamp
                if self.events
                else None
            ),
        }