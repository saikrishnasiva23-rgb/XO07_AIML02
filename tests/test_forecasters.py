import math

import pytest

from src.forecasters import (
    DynamicEnsemble,
    EWMAForecaster,
    NaiveForecaster,
    OnlineLinearForecaster,
    SeasonalNaiveForecaster,
)


def test_naive_forecaster_fit_and_predict():
    model = NaiveForecaster()
    history = [10, 12, 15, 18]

    model.fit(history)

    assert model.last_value == pytest.approx(18.0)
    assert model.predict(history) == pytest.approx(18.0)
    assert model.predict([]) == pytest.approx(0.0)


def test_seasonal_naive_forecaster_respects_season_length():
    model = SeasonalNaiveForecaster(season_length=3)
    history = [10, 12, 15, 18, 22]

    assert model.predict(history) == pytest.approx(15.0)

    short_model = SeasonalNaiveForecaster(season_length=7)
    short_history = [10, 12, 15, 18]
    assert short_model.predict(short_history) == pytest.approx(18.0)

    with pytest.raises(ValueError):
        SeasonalNaiveForecaster(season_length=0)


def test_ewma_forecaster_fit_and_predict():
    model = EWMAForecaster(alpha=0.5)
    history = [1, 3, 5]

    model.fit(history)
    assert model.level == pytest.approx(3.5)

    prediction = model.predict(history)
    assert math.isfinite(prediction)
    assert prediction == pytest.approx(3.5)

    with pytest.raises(ValueError):
        EWMAForecaster(alpha=0)

    with pytest.raises(ValueError):
        EWMAForecaster(alpha=1.5)


def test_online_linear_forecaster_training_and_prediction():
    model = OnlineLinearForecaster(lags=3)
    history = [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

    model.fit(history)
    assert model.is_fitted is True

    prediction = model.predict(history)
    assert math.isfinite(prediction)

    fallback_model = OnlineLinearForecaster(lags=4)
    assert fallback_model.predict([1, 2, 3]) == pytest.approx(3.0)


def test_dynamic_ensemble_initialization_and_forecast():
    ensemble = DynamicEnsemble(season_length=3, error_window=10)

    assert set(ensemble.models) == {"Naive", "Seasonal Naive", "EWMA", "Online Linear"}

    history = [100, 102, 101, 103, 99, 104, 102, 101, 100, 103, 105, 104, 106, 103, 101]
    ensemble.fit(history)

    predictions = ensemble.predict_all(history)
    assert set(predictions) == set(ensemble.models)
    assert all(math.isfinite(value) for value in predictions.values())

    forecast_result = ensemble.forecast(history)
    assert math.isfinite(forecast_result["forecast"])
    assert set(forecast_result["model_predictions"]) == set(ensemble.models)
    assert set(forecast_result["model_weights"]) == set(ensemble.models)
    assert all(math.isfinite(weight) for weight in forecast_result["model_weights"].values())
    assert sum(forecast_result["model_weights"].values()) == pytest.approx(1.0)

    observed = ensemble.observe(actual=110.0, values=history + [110.0])
    assert set(observed["model_errors"]) == set(ensemble.models)
    assert all(math.isfinite(error) for error in observed["model_errors"].values())

    status = ensemble.get_model_status()
    assert set(status) == set(ensemble.models)
    assert all(math.isfinite(item["recent_error"]) for item in status.values())

    ensemble.reset_weights()
    for model_name, performance in ensemble.performance.items():
        assert performance.weight == pytest.approx(0.25, rel=1e-6)

    recent_window = history[-10:]
    ensemble.retrain_recent_window(recent_window, window_size=10)
    assert ensemble.history == recent_window
    assert ensemble.last_predictions == {}


def test_no_lookahead_forecast_uses_only_supplied_values():
    values = [10, 12, 11, 13, 12, 14, 15]
    naive = NaiveForecaster()
    naive.fit(values)

    assert naive.predict(values) == pytest.approx(values[-1])

    ensemble = DynamicEnsemble()
    ensemble.fit(values)
    result = ensemble.forecast(values)

    assert math.isfinite(result["forecast"])
    assert result["forecast"] >= 0.0


@pytest.mark.parametrize(
    "values",
    [
        [],
        [5],
        [10, 10, 10, 10, 10],
        [10, 12, 14, 16, 18],
        [20, 18, 22, 19, 21, 17, 24, 16, 23, 20],
    ],
)
def test_forecasters_handle_edge_history(values):
    naive = NaiveForecaster()
    if values:
        naive.fit(values)
        assert naive.predict(values) == pytest.approx(values[-1])
    else:
        assert naive.predict(values) == pytest.approx(0.0)

    ewma = EWMAForecaster(alpha=0.3)
    if values:
        prediction = ewma.predict(values)
        assert math.isfinite(prediction)
    else:
        assert ewma.predict(values) == pytest.approx(0.0)

    seasonal = SeasonalNaiveForecaster(season_length=3)
    if values:
        prediction = seasonal.predict(values)
        assert math.isfinite(prediction)
    else:
        assert seasonal.predict(values) == pytest.approx(0.0)

    ensemble = DynamicEnsemble(season_length=3)
    result = ensemble.forecast(values)
    assert math.isfinite(result["forecast"])
    assert all(math.isfinite(weight) for weight in result["model_weights"].values())
    assert sum(result["model_weights"].values()) == pytest.approx(1.0)
