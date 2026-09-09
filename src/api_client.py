"""Evaluation client and sequential runner for the adaptive retail backend.

This module preserves the existing HTTP client while adding the missing explicit
evaluation workflow required for SC1/SC2 and offline mock execution.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Mapping, Optional

import numpy as np
import requests

from .database import RetailDatabase
from .monitor import DriftMonitor
from .system import AdaptiveRetailSystem


class EvaluationAPIError(RuntimeError):
    """Raised when the evaluation API request fails in a structured way."""


class EvaluationAPIClient:
    """Client for communicating with a retail evaluation server."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 10.0,
        headers: Optional[Mapping[str, str]] = None,
        session: Optional[requests.Session] = None,
        logger: Optional[logging.Logger] = None,
        observation_endpoint: str = "/observations",
        evaluation_endpoint: str = "/evaluation",
        health_endpoint: str = "/health",
        session_start_endpoint: str = "/session/start",
        prediction_endpoint: str = "/predictions",
    ):
        self.base_url = (
            base_url
            or os.getenv("EVALUATION_API_BASE_URL")
            or os.getenv("PS02_EVALUATION_URL")
            or ""
        ).rstrip("/")

        self.api_key = (
            api_key
            or os.getenv("EVALUATION_API_KEY")
            or os.getenv("PS02_API_KEY")
            or os.getenv("API_KEY")
        )

        self.timeout = float(timeout)
        self.observation_endpoint = observation_endpoint
        self.evaluation_endpoint = evaluation_endpoint
        self.health_endpoint = health_endpoint
        self.session_start_endpoint = session_start_endpoint
        self.prediction_endpoint = prediction_endpoint

        self.session = session or requests.Session()
        self.logger = logger or logging.getLogger(__name__)

        self.headers: Dict[str, str] = {
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        if headers:
            self.headers.update({str(k): str(v) for k, v in headers.items()})

        if self.api_key:
            self.headers["Authorization"] = f"Bearer {self.api_key}"

    def __repr__(self) -> str:
        return (
            f"EvaluationAPIClient(base_url={self.base_url!r}, "
            f"timeout={self.timeout}, headers={sorted(self.headers.keys())})"
        )

    def _build_url(self, endpoint: str) -> str:
        if not self.base_url:
            raise ValueError(
                "Evaluation API base URL is not configured. Set base_url or the "
                "EVALUATION_API_BASE_URL / PS02_EVALUATION_URL environment variable."
            )
        return f"{self.base_url}{endpoint if endpoint.startswith('/') else '/' + endpoint}"

    def _normalise_response(self, payload: Any, status_code: Optional[int] = None) -> Dict[str, Any]:
        if payload is None:
            payload = {}

        data = payload if isinstance(payload, dict) else {"data": payload}
        return {
            "ok": bool(data.get("ok", True)),
            "status_code": status_code,
            "data": data.get("data", data),
            "message": data.get("message"),
            "error": data.get("error"),
            "raw": payload,
        }

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        url = self._build_url(endpoint)
        request_headers = dict(self.headers)

        try:
            response = self.session.request(
                method=method.upper(),
                url=url,
                json=payload,
                params=dict(params) if params else None,
                headers=request_headers,
                timeout=self.timeout,
            )
            response.raise_for_status()

            try:
                parsed = response.json()
            except ValueError:
                return {
                    "ok": False,
                    "status_code": response.status_code,
                    "data": None,
                    "message": "Invalid JSON response from evaluation server.",
                    "error": "invalid_json",
                    "raw": None,
                }

            return self._normalise_response(parsed, response.status_code)

        except requests.exceptions.Timeout:
            self.logger.warning("Evaluation API request timed out for %s", url)
            return {
                "ok": False,
                "status_code": None,
                "data": None,
                "message": "Request timed out while contacting the evaluation server.",
                "error": "timeout",
                "raw": None,
            }

        except requests.exceptions.ConnectionError:
            self.logger.warning("Connection error while contacting evaluation server at %s", url)
            return {
                "ok": False,
                "status_code": None,
                "data": None,
                "message": "Connection error while contacting the evaluation server.",
                "error": "connection_error",
                "raw": None,
            }

        except requests.exceptions.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else None
            self.logger.warning("HTTP error from evaluation API: %s", status_code)
            try:
                parsed = exc.response.json() if exc.response is not None else None
            except ValueError:
                parsed = None
            return self._normalise_response(parsed or {"ok": False, "error": "http_error"}, status_code)

        except requests.exceptions.RequestException as exc:
            self.logger.warning("Request exception while contacting evaluation server: %s", exc)
            return {
                "ok": False,
                "status_code": None,
                "data": None,
                "message": str(exc),
                "error": "request_exception",
                "raw": None,
            }

    def start_session(self, stream_name: str, team_name: str, product_id: Optional[str] = None) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"stream": str(stream_name).upper(), "team_name": str(team_name)}
        if product_id is not None:
            payload["product_id"] = str(product_id)
        result = self._request("POST", self.session_start_endpoint, payload=payload)
        if not result.get("ok"):
            raise EvaluationAPIError(result.get("message") or "Could not start evaluation session.")
        session_data = result.get("data") if isinstance(result.get("data"), dict) else result
        return session_data

    def get_next_evaluation(self, product_id: Optional[str] = None, session_id: Optional[str] = None) -> Dict[str, Any]:
        params: Dict[str, Any] = {}
        if product_id is not None:
            params["product_id"] = str(product_id)
        if session_id is not None:
            params["session_id"] = str(session_id)
        return self._request("GET", self.evaluation_endpoint, params=params)

    def send_observation(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        if not isinstance(observation, Mapping):
            raise TypeError("observation must be a mapping/dictionary.")

        payload = dict(observation)
        if not payload:
            raise ValueError("observation cannot be empty.")

        required_fields = {"product_id", "timestamp"}
        missing_fields = sorted(required_fields - set(payload.keys()))
        if missing_fields:
            raise ValueError(f"Observation is missing required fields: {missing_fields}")

        result = self._request("POST", self.observation_endpoint, payload=payload)
        if result.get("ok") is False and "error" in result:
            self.logger.warning("Observation submission failed: %s", result.get("message"))
        return result

    def submit_prediction(
        self,
        session_id: str,
        prediction: float,
        row_id: Optional[str] = None,
        product_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "session_id": str(session_id),
            "prediction": float(prediction),
        }
        if row_id is not None:
            payload["row_id"] = str(row_id)
        if product_id is not None:
            payload["product_id"] = str(product_id)
        result = self._request("POST", self.prediction_endpoint, payload=payload)
        if not result.get("ok"):
            raise EvaluationAPIError(result.get("message") or "Prediction submission failed.")
        return result.get("data") if isinstance(result.get("data"), dict) else result

    def check_health(self) -> Dict[str, Any]:
        return self._request("GET", self.health_endpoint)

    def get_status(self) -> Dict[str, Any]:
        return self.check_health()


class MockEvaluationServer:
    """Offline evaluation server that simulates the sequential SC1/SC2 flow without real API calls."""

    def __init__(self, stream_name: str = "SC1", team_name: str = "demo-team", session_id: Optional[str] = None):
        self.stream_name = str(stream_name).upper()
        self.team_name = str(team_name)
        self.session_id = session_id or f"mock-{self.stream_name.lower()}-{uuid.uuid4().hex[:8]}"
        self._rows = self._build_rows()
        self._index = 0
        self._pending: Optional[Dict[str, Any]] = None

    def _build_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        base_demand = [118, 121, 124, 119, 128, 130, 127, 133, 136, 131, 145, 149, 152, 148, 160, 166, 163, 170, 178, 182]
        if self.stream_name == "SC2":
            base_demand = [110, 109, 108, 112, 116, 121, 124, 120, 130, 138, 148, 164, 176, 190, 210, 200, 188, 182, 176, 170]

        for idx, actual in enumerate(base_demand):
            product = "fresh-bread" if idx % 2 == 0 else "fresh-sandwich"
            row = {
                "row_id": f"{self.stream_name.lower()}-{idx}",
                "product_id": product,
                "product_name": "Fresh Bread" if product == "fresh-bread" else "Fresh Sandwich",
                "timestamp": (datetime(2024, 1, 1) + timedelta(days=idx)).isoformat(timespec="seconds"),
                "date": (datetime(2024, 1, 1) + timedelta(days=idx)).strftime("%Y-%m-%d"),
                "stock_purchased": float(120 + idx * 2),
                "units_sold": float(actual),
                "actual_demand": float(actual),
                "event_info": "normal" if idx < len(base_demand) - 4 else "promo",
                "fiscal_year": "FY-2024",
                "features": {
                    "day_of_week": idx % 7,
                    "trend": float(idx * 1.5),
                    "seasonality": float(10 * np.sin(idx / 3.0)),
                    "temperature_index": float(20 + idx % 5),
                    "event_signal": 1.0 if idx > 12 else 0.0,
                },
                "target": None,
                "actual": float(actual),
            }
            rows.append(row)
        return rows

    def start_session(self, stream_name: Optional[str] = None, team_name: Optional[str] = None) -> Dict[str, Any]:
        if stream_name is not None:
            self.stream_name = str(stream_name).upper()
        if team_name is not None:
            self.team_name = str(team_name)
        return {
            "session_id": self.session_id,
            "stream": self.stream_name,
            "team_name": self.team_name,
            "status": "active",
        }

    def get_next_row(self) -> Dict[str, Any]:
        if self._index >= len(self._rows):
            return {"done": True, "session_id": self.session_id, "stream": self.stream_name, "row": None}

        row = self._rows[self._index]
        self._pending = {"row_id": row["row_id"], "product_id": row["product_id"], "actual": row["actual"]}
        self._index += 1

        payload = dict(row)
        payload.pop("actual", None)
        payload["target"] = None
        payload["session_id"] = self.session_id
        payload["stream"] = self.stream_name
        payload["done"] = False

        return {"session_id": self.session_id, "stream": self.stream_name, "row": payload, "target": None, "done": False}

    def submit_prediction(self, prediction: float, row_id: Optional[str] = None, product_id: Optional[str] = None) -> Dict[str, Any]:
        if self._pending is None:
            raise ValueError("No pending row is available for prediction submission.")

        row_id_value = row_id or self._pending["row_id"]
        product_id_value = product_id or self._pending["product_id"]
        actual = float(self._pending["actual"])
        predicted = float(prediction)
        outcome = {
            "session_id": self.session_id,
            "stream": self.stream_name,
            "row_id": row_id_value,
            "product_id": product_id_value,
            "prediction": predicted,
            "actual": actual,
            "error": abs(actual - predicted),
            "target_revealed": True,
            "done": False,
        }
        self._pending = None
        return outcome


class EvaluationRunner:
    """Explicit, no-lookahead evaluation runner built on the existing forecasting engine."""

    def __init__(
        self,
        server: Optional[Any] = None,
        stream_name: str = "SC1",
        team_name: str = "demo-team",
        product_id: str = "fresh-bread",
        db_path: Optional[str] = None,
    ):
        self.stream_name = str(stream_name).upper()
        self.team_name = str(team_name)
        self.product_id = str(product_id)
        self.server = server or MockEvaluationServer(stream_name=self.stream_name, team_name=self.team_name)
        self.session_id = getattr(self.server, "session_id", None)
        self.db = RetailDatabase(db_path=db_path)
        self._ensure_product_catalog()
        self.monitor = DriftMonitor(reference_window=12, detection_window=4, persistence_required=2)
        self.system = AdaptiveRetailSystem(reference_window=12, detection_window=4, persistence_required=2)
        self.history: List[Dict[str, Any]] = []
        self._last_row: Optional[Dict[str, Any]] = None
        self._initialized_history: List[float] = self._read_product_history()
        if self._initialized_history:
            self.system.initialize(self._initialized_history)

    def _ensure_product_catalog(self) -> None:
        existing = self.db.get_product_catalog(include_perishable_only=True)
        names = {row["product_id"] for row in existing}
        if "fresh-bread" not in names:
            self.db.add_product(
                "fresh-bread",
                "Fresh Bread",
                category="Perishable Food",
                current_stock=120.0,
                perishable=True,
                perishable_lifetime_days=1.0,
            )
        if "fresh-sandwich" not in names:
            self.db.add_product(
                "fresh-sandwich",
                "Fresh Sandwich",
                category="Perishable Food",
                current_stock=90.0,
                perishable=True,
                perishable_lifetime_days=1.0,
            )

    def _read_product_history(self) -> List[float]:
        rows = self.db.get_product_history(self.product_id)
        if rows:
            return [float(item.get("demand", item.get("actual_demand", 0.0))) for item in rows]
        return []

    def _generate_prediction(self, features: Mapping[str, Any]) -> float:
        history = self._read_product_history()
        if not history:
            baseline = [118.0, 121.0, 120.0, 124.0, 119.0, 123.0, 128.0, 125.0]
            history = baseline

        if not self.system.initialized:
            self.system.initialize(history)
        prediction_payload = self.system.forecast()
        prediction = float(prediction_payload["forecast"])

        if "features" in features:
            feature_payload = features["features"]
            if isinstance(feature_payload, Mapping) and feature_payload.get("event_signal"):
                prediction *= 1.12

        return float(max(0.0, prediction))

    def _record_demand(self, features: Mapping[str, Any], actual: float, prediction: float, error: float) -> Dict[str, Any]:
        self.db.record_demand(
            product_id=self.product_id,
            demand=float(actual),
            timestamp=str(features.get("timestamp") or datetime.now().isoformat(timespec="seconds")),
            stock_purchased=float(features.get("stock_purchased", 0.0) or 0.0),
            units_sold=float(features.get("units_sold", actual) or actual),
            previous_prediction=float(prediction),
            prediction_error=float(error),
            event_info=str(features.get("event_info") or "normal"),
            fiscal_year=str(features.get("fiscal_year") or "FY-2024"),
        )

        if self.system.initialized:
            result = self.system.process_observation(actual)
            raw_status = result["observation"]["system_status"]
            drift_data = result["observation"]["drift"]
            if raw_status == "ADAPTING":
                monitor_status = "PERSISTENT CHANGE" if drift_data.get("persistent_change") else "WATCH"
            elif raw_status in {"WARMING UP", "WARMING_UP"}:
                monitor_status = "STABLE"
            elif raw_status in {"PERSISTENT CHANGE", "POSSIBLE CHANGE", "WATCH", "STABLE"}:
                monitor_status = raw_status
            else:
                monitor_status = "STABLE"
        else:
            monitor_status = "STABLE"
            drift_data = {"persistent_change": False, "change_detected": False, "score": 0.0}

        return {
            "monitor_status": monitor_status,
            "drift_score": drift_data.get("score", drift_data.get("drift_score", 0.0)),
            "persistent_change": bool(drift_data.get("persistent_change", False)),
            "adaptation_triggered": bool(result["observation"]["adaptation"]["triggered"]) if self.system.initialized else False,
        }

    def start_session(self) -> Dict[str, Any]:
        session = self.server.start_session(stream_name=self.stream_name, team_name=self.team_name)
        self.session_id = session.get("session_id")
        return session

    def get_status(self) -> Dict[str, Any]:
        return {
            "stream": self.stream_name,
            "team_name": self.team_name,
            "session_id": self.session_id,
            "rows_processed": len(self.history),
            "current_product": self.product_id,
            "initialized": self.system.initialized,
        }

    def run(self, max_rows: int = 10) -> Dict[str, Any]:
        if self.session_id is None:
            self.start_session()

        processed = 0
        while processed < max_rows:
            row_packet = self.server.get_next_row()
            if row_packet.get("done"):
                break

            row = row_packet.get("row", {})
            if not row:
                break

            target_is_hidden = row.get("target") is None
            if not target_is_hidden:
                raise ValueError("Target leakage: actual target was revealed before prediction generation.")

            features = row.get("features", row)
            prediction = self._generate_prediction(row)
            submission = self.server.submit_prediction(prediction, row_id=row.get("row_id"), product_id=self.product_id)
            actual = float(submission["actual"])
            error = float(submission["error"])

            monitoring = self._record_demand(row, actual, prediction, error)

            self.history.append({
                "row_id": row.get("row_id"),
                "product_id": self.product_id,
                "prediction": float(prediction),
                "actual": float(actual),
                "error": float(error),
                "monitor_status": monitoring["monitor_status"],
                "persistent_change": monitoring["persistent_change"],
                "adaptation_triggered": monitoring["adaptation_triggered"],
                "used_target_before_prediction": not target_is_hidden,
            })
            processed += 1

        return {
            "stream": self.stream_name,
            "rows_processed": len(self.history),
            "session_id": self.session_id,
            "history": self.history,
        }


__all__ = [
    "EvaluationAPIClient",
    "EvaluationAPIError",
    "MockEvaluationServer",
    "EvaluationRunner",
]
