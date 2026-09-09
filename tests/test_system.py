import math

import pytest

from src.system import AdaptiveRetailSystem


def _stable_history(length: int = 50, center: float = 100.0):
    values = []
    for idx in range(length):
        seasonal_component = 2.0 * math.sin(idx / 5.0)
        small_drift = 0.4 * math.cos(idx / 9.0)
        values.append(center + seasonal_component + small_drift)
    return values


def _gradual_shift_history(length: int = 25, start: float = 100.0):
    return [start + idx * 1.5 for idx in range(length)]


def test_initialization_sets_up_system_and_history():
    history = _stable_history(45)
    system = AdaptiveRetailSystem()

    result = system.initialize(history)

    assert result["initialized"] is True
    assert system.initialized is True
    assert len(system.get_history()) == len(history)
    assert system.get_history() == pytest.approx(history)
    assert system.forecaster.get_model_status()
    assert system.monitor.history


def test_forecast_returns_numeric_output_and_valid_weights():
    history = _stable_history(50)
    system = AdaptiveRetailSystem()
    system.initialize(history)

    result = system.forecast()

    assert isinstance(result["forecast"], float)
    assert math.isfinite(result["forecast"])
    assert result["model_predictions"]
    assert set(result["model_predictions"]) == {"Naive", "Seasonal Naive", "EWMA", "Online Linear"}
    assert set(result["model_weights"]) == {"Naive", "Seasonal Naive", "EWMA", "Online Linear"}
    total_weight = sum(result["model_weights"].values())
    assert total_weight == pytest.approx(1.0, abs=1e-6)
    assert all(math.isfinite(value) for value in result["model_weights"].values())


def test_observe_requires_a_forecast_first():
    system = AdaptiveRetailSystem()
    system.initialize(_stable_history(45))

    with pytest.raises(RuntimeError, match="A forecast must be generated before observing actual data"):
        system.observe(120.0)


def test_process_observation_follows_no_lookahead_sequence():
    history = _stable_history(50)
    system = AdaptiveRetailSystem()
    system.initialize(history)

    forecast_before = system.forecast()
    actual = 110.0
    result = system.process_observation(actual)

    assert result["forecast"]["forecast"] == pytest.approx(forecast_before["forecast"])
    assert result["observation"]["forecast"] == pytest.approx(forecast_before["forecast"])
    assert result["observation"]["actual"] == pytest.approx(actual)
    assert result["observation"]["error"] == pytest.approx(abs(actual - forecast_before["forecast"]))
    assert system.last_forecast is None
    assert system.get_history()[-1] == pytest.approx(actual)


def test_normal_demand_remains_stable_without_unnecessary_adaptation():
    system = AdaptiveRetailSystem(reference_window=12, detection_window=4, persistence_required=3)
    system.initialize(_stable_history(30, center=100.0))

    for value in [101.2, 99.8, 100.4, 101.0, 100.3, 99.9, 100.6, 101.1, 100.0, 99.7]:
        result = system.process_observation(value)
        assert result["observation"]["adaptation"]["triggered"] is False
        assert result["observation"]["drift"]["persistent_change"] is False
        assert result["observation"]["system_status"] in {"STABLE", "WATCH", "WARMING UP", "POSSIBLE CHANGE"}


def test_temporary_anomaly_does_not_immediately_trigger_persistent_change():
    system = AdaptiveRetailSystem(reference_window=12, detection_window=4, persistence_required=3)
    system.initialize(_stable_history(35, center=100.0))

    for value in [101.0, 99.5, 100.5, 100.0, 99.8, 100.2, 101.0, 100.2, 99.9, 100.4]:
        system.process_observation(value)

    result = system.process_observation(170.0)

    assert math.isfinite(result["observation"]["drift"]["score"])
    assert 0.0 <= result["observation"]["drift"]["score"] <= 1.0
    assert result["observation"]["drift"]["persistent_change"] is False
    assert result["observation"]["adaptation"]["triggered"] is False
    assert result["observation"]["system_status"] not in {"ADAPTING", "PERSISTENT CHANGE"}


def test_persistent_change_eventually_triggers_adaptation():
    system = AdaptiveRetailSystem(
        reference_window=12,
        detection_window=4,
        persistence_required=2,
        adaptation_cooldown=1,
    )
    baseline = _stable_history(30, center=100.0)
    system.initialize(baseline)

    trigger_seen = False
    adaptation_event = None
    for value in _gradual_shift_history(length=30, start=100.0):
        result = system.process_observation(value)
        if result["observation"]["adaptation"]["triggered"]:
            trigger_seen = True
            adaptation_event = result
            break

    assert trigger_seen, "The system did not trigger adaptation under a sustained gradual shift."
    assert adaptation_event is not None
    assert adaptation_event["observation"]["adaptation"]["details"]["adapted"] is True
    assert adaptation_event["observation"]["adaptation"]["details"]["action"] == "Recent-window retraining + dynamic reweighting"
    assert len(system.get_adaptation_events()) >= 1


def test_adaptation_cooldown_prevents_repeated_adaptations():
    system = AdaptiveRetailSystem(
        reference_window=12,
        detection_window=4,
        persistence_required=2,
        adaptation_cooldown=3,
    )
    system.initialize(_stable_history(30, center=100.0))

    first_trigger = None
    for value in _gradual_shift_history(length=30, start=100.0):
        result = system.process_observation(value)
        if result["observation"]["adaptation"]["triggered"]:
            first_trigger = result
            break

    assert first_trigger is not None

    next_result = system.process_observation(150.0)
    assert next_result["observation"]["adaptation"]["triggered"] is False
    assert system.get_status()["adaptation"]["steps_since_adaptation"] >= 1


def test_status_and_history_are_consistent_after_observations():
    system = AdaptiveRetailSystem(reference_window=12, detection_window=4, persistence_required=2)
    system.initialize(_stable_history(30, center=100.0))

    for value in [101.0, 99.8, 100.5, 102.1, 101.4]:
        system.process_observation(value)

    status = system.get_status()
    assert status["initialized"] is True
    assert status["step"] > 0
    assert status["data_points"] == len(system.get_history())
    assert status["last_actual"] == pytest.approx(system.get_history()[-1])
    assert "monitor" in status and "adaptation" in status and "models" in status
    assert math.isfinite(float(status["monitor"]["drift_score"]))
    assert status["adaptation"]["adaptation_count"] >= 0


def test_adaptation_events_are_recorded_and_useful():
    system = AdaptiveRetailSystem(
        reference_window=12,
        detection_window=4,
        persistence_required=2,
        adaptation_cooldown=1,
    )
    system.initialize(_stable_history(30, center=100.0))

    seen = False
    for value in _gradual_shift_history(length=30, start=100.0):
        result = system.process_observation(value)
        if result["observation"]["adaptation"]["triggered"]:
            seen = True
            break

    assert seen
    events = system.get_adaptation_events()
    assert events
    event = events[0]
    assert "timestamp" in event
    assert "reason" in event
    assert "drift_score" in event
    assert "action" in event
    assert "data_points_used" in event
    assert event["action"] == "Recent-window retraining + dynamic reweighting"


def test_process_observation_returns_expected_structure():
    system = AdaptiveRetailSystem(reference_window=12, detection_window=4, persistence_required=2)
    history = _stable_history(35, center=100.0)
    system.initialize(history)

    result = system.process_observation(108.0)

    assert set(result) == {"forecast", "observation"}
    assert "forecast" in result["forecast"]
    assert "model_predictions" in result["forecast"]
    assert "model_weights" in result["forecast"]
    assert "actual" in result["observation"]
    assert "error" in result["observation"]
    assert "drift" in result["observation"]
    assert "adaptation" in result["observation"]
    assert "system_status" in result["observation"]


@pytest.mark.parametrize(
    "history",
    [
        [100.0, 101.0, 99.0, 100.0, 101.0],
        _stable_history(25, center=100.0),
        [100.0, 100.0, 100.0, 100.0, 100.0],
    ],
)
def test_system_handles_short_and_constant_histories_without_crashing(history):
    system = AdaptiveRetailSystem(reference_window=10, detection_window=4, persistence_required=2)
    system.initialize(history)
    forecast = system.forecast()
    assert math.isfinite(forecast["forecast"])
    result = system.process_observation(history[-1] + 2.0)
    assert math.isfinite(result["observation"]["error"])
    assert "system_status" in result["observation"]
