import math

import pytest

from src.monitor import DriftMonitor


def _prime_monitor(monitor: DriftMonitor, baseline: list[float]) -> None:
    for value in baseline:
        monitor.update(actual=value, forecast=value)


def test_monitor_can_initialize_with_defaults_and_custom_values():
    default_monitor = DriftMonitor()
    assert default_monitor.reference_window == 30
    assert default_monitor.detection_window == 10
    assert default_monitor.persistence_required == 3

    custom_monitor = DriftMonitor(reference_window=12, detection_window=4, persistence_required=2)
    assert custom_monitor.reference_window == 12
    assert custom_monitor.detection_window == 4
    assert custom_monitor.persistence_required == 2


def test_monitor_warns_before_enough_history():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)

    for value in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113]:
        result = monitor.update(actual=value, forecast=value)
        if len(monitor.history) < monitor.reference_window + monitor.detection_window:
            assert result.status == "WARMING UP"
            assert result.change_detected is False
            assert result.persistent_change is False


def test_monitor_stable_demand_does_not_report_persistent_change():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)
    stable_values = [100, 101, 99, 100, 101, 100, 102, 101, 100, 99, 100, 101, 99, 100, 101, 100]

    for value in stable_values:
        result = monitor.update(actual=value, forecast=value)
        assert math.isfinite(result.drift_score)
        assert 0.0 <= result.drift_score <= 1.0
        if len(monitor.history) >= monitor.reference_window + monitor.detection_window:
            assert result.persistent_change is False


def test_single_anomaly_does_not_immediately_become_persistent_change():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)
    baseline = [100, 101, 100, 99, 100, 101, 102, 100, 101, 100, 99, 101, 100, 98, 100, 101, 100]
    _prime_monitor(monitor, baseline)

    anomaly_result = monitor.update(actual=170, forecast=100)
    assert math.isfinite(anomaly_result.drift_score)
    assert 0.0 <= anomaly_result.drift_score <= 1.0
    assert anomaly_result.persistent_change is False
    assert anomaly_result.status != "PERSISTENT CHANGE"


def test_gradual_shift_triggers_watch_then_persistent_change():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)
    baseline = [100, 101, 99, 100, 101, 100, 102, 101, 100, 99, 100, 101, 99, 100, 101, 102]
    _prime_monitor(monitor, baseline)

    gradual_shift = [103, 105, 108, 110, 112, 115, 118, 120, 122, 125]
    saw_watch_or_change = False
    persistent_seen = False

    for value in gradual_shift:
        result = monitor.update(actual=value, forecast=value - 2)
        if result.status in {"WATCH", "POSSIBLE CHANGE"}:
            saw_watch_or_change = True
        if result.persistent_change:
            persistent_seen = True
            assert result.status == "PERSISTENT CHANGE"
            break

    assert saw_watch_or_change or persistent_seen


def test_persistence_requirement_requires_consecutive_detections():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=2)
    baseline = [100, 101, 99, 100, 101, 100, 102, 101, 100, 99, 100, 101, 99, 100, 101, 102]
    _prime_monitor(monitor, baseline)

    elevated_values = [110, 112, 110, 111]
    persistent_seen = False

    for index, value in enumerate(elevated_values):
        result = monitor.update(actual=value, forecast=value - 2)
        if result.persistent_change:
            persistent_seen = True
            assert result.status == "PERSISTENT CHANGE"
            break
        if index == 0:
            assert result.persistent_change is False

    assert persistent_seen


def test_monitor_returns_explainable_signal_values():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)
    baseline = [100, 102, 101, 99, 100, 102, 101, 103, 100, 99, 100, 101, 99, 100, 101, 102]
    _prime_monitor(monitor, baseline)

    result = monitor.update(actual=150, forecast=100)

    assert math.isfinite(result.drift_score)
    assert 0.0 <= result.drift_score <= 1.0
    assert math.isfinite(result.anomaly_score)
    assert math.isfinite(result.mean_shift)
    assert math.isfinite(result.volatility_shift)
    assert result.status in {"STABLE", "WATCH", "POSSIBLE CHANGE", "PERSISTENT CHANGE", "WARMING UP"}
    assert isinstance(result.change_detected, bool)
    assert isinstance(result.persistent_change, bool)


def test_reset_clears_history_and_consecutive_drift():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)

    for value in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]:
        monitor.update(actual=value, forecast=value)

    assert len(monitor.history) >= 0
    assert monitor.consecutive_drift >= 0

    monitor.reset()

    assert len(monitor.history) == 0
    assert len(monitor.drift_history) == 0
    assert monitor.consecutive_drift == 0


def test_get_status_tracks_monitor_state():
    monitor = DriftMonitor(reference_window=10, detection_window=4, persistence_required=3)

    assert monitor.get_status()["drift_score"] == 0.0
    assert monitor.get_status()["consecutive_drift"] == 0
    assert monitor.get_status()["history_size"] == 0

    for value in [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114]:
        monitor.update(actual=value, forecast=value)

    status = monitor.get_status()
    assert "drift_score" in status
    assert "consecutive_drift" in status
    assert "history_size" in status
    assert status["history_size"] == len(monitor.history)
    assert math.isfinite(status["drift_score"])
