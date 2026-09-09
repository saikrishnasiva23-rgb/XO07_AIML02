import math
import tempfile
from pathlib import Path

import pytest

from src.api_client import EvaluationRunner, MockEvaluationServer
from src.database import RetailDatabase
from src.spark_loader import ElectricityDataLoader
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


def test_empty_electricity_dataset_raises_clear_error(tmp_path):
    csv_path = tmp_path / "electricity.csv"
    csv_path.write_text("")

    loader = ElectricityDataLoader(data_path=csv_path)
    try:
        with pytest.raises(ValueError, match="empty|missing required columns"):
            loader.load_dataset()
    finally:
        loader.close()


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


def test_perishable_product_catalog_and_fiscal_history_support():
    with tempfile.TemporaryDirectory() as tmpdir:
        db = RetailDatabase(db_path=str(Path(tmpdir) / "product_test.db"))

        db.add_product("fresh-bread", "Fresh Bread", category="Perishable Food", current_stock=120.0)
        db.add_product("fresh-sandwich", "Fresh Sandwich", category="Perishable Food", current_stock=90.0)
        db.add_product("prepared-salad", "Prepared Salad", category="Perishable Food", current_stock=60.0)

        catalog = db.get_product_catalog(include_perishable_only=True)
        assert {row["product_name"] for row in catalog} >= {"Fresh Bread", "Fresh Sandwich", "Prepared Salad"}

        history_2023 = [
            {"timestamp": "2023-01-01T00:00:00", "demand": 100.0, "stock_purchased": 110.0, "units_sold": 100.0},
            {"timestamp": "2023-02-01T00:00:00", "demand": 120.0, "stock_purchased": 125.0, "units_sold": 120.0},
            {"timestamp": "2023-03-01T00:00:00", "demand": 130.0, "stock_purchased": 140.0, "units_sold": 130.0},
        ]
        history_2024 = [
            {"timestamp": "2024-01-01T00:00:00", "demand": 150.0, "stock_purchased": 160.0, "units_sold": 150.0},
            {"timestamp": "2024-02-01T00:00:00", "demand": 170.0, "stock_purchased": 175.0, "units_sold": 170.0},
        ]
        for row in history_2023 + history_2024:
            db.record_demand(
                product_id="fresh-bread",
                demand=row["demand"],
                stock_purchased=row["stock_purchased"],
                units_sold=row["units_sold"],
                timestamp=row["timestamp"],
            )

        product_history = db.get_product_history("fresh-bread")
        assert len(product_history) >= 5
        assert product_history[-1]["demand"] == pytest.approx(170.0)

        previous_year = db.get_previous_fiscal_year_summary("fresh-bread")
        assert previous_year["product_id"] == "fresh-bread"
        assert previous_year["total_demand"] >= 350.0
        assert previous_year["total_stock_purchased"] >= 350.0
        assert previous_year["monthly_demand"]

        context = db.get_product_context("fresh-bread")
        assert context["product_name"] == "Fresh Bread"
        assert context["history"]
        assert "previous_fiscal_year" in context

        db.close()


def test_mock_sc1_runner_runs_sequential_prediction_cycle_without_target_leakage():
    mock_server = MockEvaluationServer(stream_name="SC1")
    runner = EvaluationRunner(server=mock_server, stream_name="SC1", team_name="demo-team")

    result = runner.run(max_rows=4)

    assert result["stream"] == "SC1"
    assert result["rows_processed"] == 4
    assert result["session_id"] == mock_server.session_id
    assert result["history"]
    assert all(item["prediction"] is not None for item in result["history"])
    assert all(item["actual"] is not None for item in result["history"])
    assert all(item["error"] is not None for item in result["history"])
    assert all(item["used_target_before_prediction"] is False for item in result["history"])


def test_mock_sc2_runner_keeps_session_state_and_tracks_monitoring():
    mock_server = MockEvaluationServer(stream_name="SC2")
    runner = EvaluationRunner(server=mock_server, stream_name="SC2", team_name="demo-team")

    result = runner.run(max_rows=3)

    assert result["stream"] == "SC2"
    assert result["rows_processed"] == 3
    assert result["session_id"] == mock_server.session_id
    assert all(item["monitor_status"] in {"STABLE", "WATCH", "POSSIBLE CHANGE", "PERSISTENT CHANGE"} for item in result["history"])
    assert "session_id" in runner.get_status()
