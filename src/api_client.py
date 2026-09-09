"""Lightweight HTTP client for communicating with the evaluation server.

This module is intentionally limited to external API communication. It does not
perform forecasting, adaption, or drift detection. Those responsibilities remain
with the project's existing forecasting, monitoring, and adaptation modules.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Mapping, Optional

import requests


class EvaluationAPIError(RuntimeError):
    """Raised when the evaluation API request fails in a structured way."""


class EvaluationAPIClient:
    """Client for communicating with a retail evaluation server.

    The actual endpoint is provided through the constructor or an environment
    variable such as EVALUATION_API_BASE_URL or PS02_EVALUATION_URL. No real
    credentials are embedded in the source code.
    """

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
        """Return a safe representation without exposing credentials."""

        return (
            f"EvaluationAPIClient(base_url={self.base_url!r}, "
            f"timeout={self.timeout}, headers={sorted(self.headers.keys())})"
        )

    def _build_url(self, endpoint: str) -> str:
        """Construct a request URL from the configured base URL and endpoint."""

        if not self.base_url:
            raise ValueError(
                "Evaluation API base URL is not configured. Set base_url or the "
                "EVALUATION_API_BASE_URL / PS02_EVALUATION_URL environment variable."
            )
        return f"{self.base_url}{endpoint if endpoint.startswith('/') else '/' + endpoint}"

    def _normalise_response(self, payload: Any, status_code: Optional[int] = None) -> Dict[str, Any]:
        """Convert raw server payloads into a predictable dictionary."""

        if payload is None:
            payload = {}

        data = payload if isinstance(payload, dict) else {"data": payload}
        response = {
            "ok": bool(data.get("ok", True)),
            "status_code": status_code,
            "data": data.get("data", data),
            "message": data.get("message"),
            "error": data.get("error"),
            "raw": payload,
        }

        return response

    def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[Mapping[str, Any]] = None,
        params: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a request with safe handling for HTTP, JSON, and connectivity failures."""

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

    def send_observation(self, observation: Mapping[str, Any]) -> Dict[str, Any]:
        """Send a retail demand observation to the evaluation server.

        Example payload:
        {
            "product_id": "SKU-100",
            "timestamp": "2026-09-09T12:00:00",
            "actual_demand": 120.0,
            "forecast": 118.5,
            "forecast_error": 1.5,
            "drift_score": 0.24,
            "adaptation_status": "STABLE",
        }
        """

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

    def get_next_evaluation(self, product_id: Optional[str] = None) -> Dict[str, Any]:
        """Retrieve the next available evaluation payload from the server."""

        params: Dict[str, Any] = {}
        if product_id is not None:
            params["product_id"] = str(product_id)

        return self._request("GET", self.evaluation_endpoint, params=params)

    def check_health(self) -> Dict[str, Any]:
        """Check whether the evaluation server is reachable and healthy."""

        return self._request("GET", self.health_endpoint)

    def get_status(self) -> Dict[str, Any]:
        """Alias for a health-status check for dashboard or caller convenience."""

        return self.check_health()


__all__ = ["EvaluationAPIClient", "EvaluationAPIError"]
