"""Inventory planning utilities built on top of the adaptive retail forecasting system.

This module intentionally reuses the existing adaptive demand forecast produced by
:class:`src.system.AdaptiveRetailSystem` rather than introducing a separate
forecasting model. Each product is evaluated by:

1. generating an adaptive demand forecast from its recent historical demand,
2. estimating demand volatility from the same history,
3. calculating replenishment metrics such as safety stock and reorder point, and
4. classifying the inventory state as healthy, replenishment, critical, or
   overstocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

import numpy as np

from .system import AdaptiveRetailSystem


@dataclass
class InventoryConfig:
    """Configuration parameters for the inventory control logic."""

    lead_time: float = 7.0
    safety_stock_factor: float = 1.0
    service_level: float = 0.95
    review_period: float = 7.0

    def __post_init__(self) -> None:
        if self.lead_time <= 0:
            raise ValueError("lead_time must be greater than zero.")
        if self.safety_stock_factor < 0:
            raise ValueError("safety_stock_factor must be non-negative.")
        if not 0 < self.service_level < 1:
            raise ValueError("service_level must be between 0 and 1.")
        if self.review_period <= 0:
            raise ValueError("review_period must be greater than zero.")


@dataclass
class InventoryRecommendation:
    """Complete recommendation for one product's inventory position."""

    product_id: str
    current_stock: float
    forecasted_demand: float
    safety_stock: float
    reorder_point: float
    recommended_reorder_quantity: float
    stock_out_risk: float
    excess_inventory_risk: float
    inventory_status: str
    lead_time: float
    review_period: float
    service_level: float

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the recommendation into a plain dictionary."""

        return {
            "product_id": self.product_id,
            "current_stock": float(self.current_stock),
            "forecasted_demand": float(self.forecasted_demand),
            "safety_stock": float(self.safety_stock),
            "reorder_point": float(self.reorder_point),
            "recommended_reorder_quantity": float(self.recommended_reorder_quantity),
            "stock_out_risk": float(self.stock_out_risk),
            "excess_inventory_risk": float(self.excess_inventory_risk),
            "inventory_status": self.inventory_status,
            "lead_time": float(self.lead_time),
            "review_period": float(self.review_period),
            "service_level": float(self.service_level),
        }


class InventoryManager:
    """Generate adaptive replenishment recommendations for retail products."""

    def __init__(self, config: Optional[InventoryConfig] = None):
        self.config = config or InventoryConfig()

    def _as_float_array(self, values: Sequence[float]) -> np.ndarray:
        """Normalize values to a float NumPy array."""

        arr = np.asarray(list(values), dtype=float)
        if arr.size == 0:
            raise ValueError("Historical demand cannot be empty.")
        return arr

    def _forecast_demand(self, historical_demand: Sequence[float]) -> float:
        """Generate the adaptive forecast for a product historical demand series."""

        values = self._as_float_array(historical_demand)

        system = AdaptiveRetailSystem()
        system.initialize(values.tolist())
        forecast = system.forecast()

        return float(forecast["forecast"])

    def _demand_std_dev(self, historical_demand: Sequence[float]) -> float:
        """Estimate demand volatility from historical demand observations."""

        values = self._as_float_array(historical_demand)

        if values.size <= 1:
            return 0.0

        return float(np.std(values, ddof=1))

    @staticmethod
    def _clip(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
        """Clip a float into a bounded range."""

        return float(max(lower, min(upper, value)))

    def calculate_safety_stock(
        self,
        forecasted_demand: float,
        demand_std_dev: float,
        lead_time: Optional[float] = None,
        safety_stock_factor: Optional[float] = None,
        service_level: Optional[float] = None,
    ) -> float:
        """Calculate safety stock using a service-level based normal approximation."""

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        _factor = (
            self.config.safety_stock_factor
            if safety_stock_factor is None
            else float(safety_stock_factor)
        )
        _service_level = (
            self.config.service_level
            if service_level is None
            else float(service_level)
        )

        if _service_level <= 0 or _service_level >= 1:
            raise ValueError("service_level must be strictly between 0 and 1.")

        z_score = NormalDist().inv_cdf(_service_level)
        variability = max(float(demand_std_dev), 0.0)
        safety_stock = _factor * z_score * variability * np.sqrt(max(_lead_time, 1.0))
        return max(0.0, float(safety_stock))

    def calculate_reorder_point(
        self,
        forecasted_demand: float,
        demand_std_dev: float,
        lead_time: Optional[float] = None,
        safety_stock_factor: Optional[float] = None,
        service_level: Optional[float] = None,
    ) -> float:
        """Calculate the reorder point on the basis of lead-time demand plus safety stock."""

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        avg_demand = max(float(forecasted_demand), 0.0)
        safety_stock = self.calculate_safety_stock(
            forecasted_demand=avg_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            safety_stock_factor=safety_stock_factor,
            service_level=service_level,
        )
        reorder_point = avg_demand * _lead_time + safety_stock
        return max(0.0, float(reorder_point))

    def calculate_recommended_reorder_quantity(
        self,
        current_stock: float,
        forecasted_demand: float,
        demand_std_dev: float,
        lead_time: Optional[float] = None,
        review_period: Optional[float] = None,
        safety_stock_factor: Optional[float] = None,
        service_level: Optional[float] = None,
    ) -> float:
        """Calculate the recommended quantity to reorder from the target stock level."""

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        _review_period = self.config.review_period if review_period is None else float(review_period)
        safety_stock = self.calculate_safety_stock(
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            safety_stock_factor=safety_stock_factor,
            service_level=service_level,
        )
        target_inventory = (
            max(float(forecasted_demand), 0.0) * (_lead_time + _review_period)
            + safety_stock
        )
        recommended_order = target_inventory - max(float(current_stock), 0.0)
        return max(0.0, float(recommended_order))

    def estimate_stock_out_risk(
        self,
        current_stock: float,
        forecasted_demand: float,
        demand_std_dev: float,
        lead_time: Optional[float] = None,
    ) -> float:
        """Estimate the probability that demand during lead time exceeds available stock."""

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        avg_demand = max(float(forecasted_demand), 0.0)
        volatility = max(float(demand_std_dev), 0.0)
        lead_time_demand = avg_demand * _lead_time
        lead_time_volatility = volatility * np.sqrt(max(_lead_time, 1.0))

        if lead_time_volatility <= 0:
            return 0.0 if float(current_stock) >= lead_time_demand else 1.0

        z_score = (float(current_stock) - lead_time_demand) / lead_time_volatility
        cdf_value = NormalDist().cdf(z_score)
        risk = 1.0 - cdf_value
        return self._clip(float(risk), 0.0, 1.0)

    def detect_excess_inventory(
        self,
        current_stock: float,
        forecasted_demand: float,
        demand_std_dev: float,
        lead_time: Optional[float] = None,
        review_period: Optional[float] = None,
        safety_stock_factor: Optional[float] = None,
        service_level: Optional[float] = None,
    ) -> float:
        """Estimate the risk that current inventory is above the target replenishment level."""

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        _review_period = self.config.review_period if review_period is None else float(review_period)
        safety_stock = self.calculate_safety_stock(
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            safety_stock_factor=safety_stock_factor,
            service_level=service_level,
        )
        target_inventory = (
            max(float(forecasted_demand), 0.0) * (_lead_time + _review_period)
            + safety_stock
        )
        excess_gap = max(float(current_stock) - target_inventory, 0.0)

        if excess_gap <= 0:
            return 0.0

        scale = max(float(forecasted_demand) * max(_review_period, 1.0), 1.0)
        normalized_risk = excess_gap / scale
        return self._clip(normalized_risk / 1.5, 0.0, 1.0)

    def generate_inventory_status(
        self,
        current_stock: float,
        forecasted_demand: float,
        demand_std_dev: float,
        lead_time: Optional[float] = None,
        review_period: Optional[float] = None,
        safety_stock_factor: Optional[float] = None,
        service_level: Optional[float] = None,
    ) -> str:
        """Classify an item into a replenishment or risk status."""

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        _review_period = self.config.review_period if review_period is None else float(review_period)

        reorder_point = self.calculate_reorder_point(
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            safety_stock_factor=safety_stock_factor,
            service_level=service_level,
        )
        stock_out_risk = self.estimate_stock_out_risk(
            current_stock=current_stock,
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
        )
        excess_risk = self.detect_excess_inventory(
            current_stock=current_stock,
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            review_period=_review_period,
            safety_stock_factor=safety_stock_factor,
            service_level=service_level,
        )

        if current_stock <= reorder_point * 0.8 or stock_out_risk >= 0.6:
            return "Critical"
        if current_stock <= reorder_point or stock_out_risk >= 0.35:
            return "Replenish"
        if excess_risk >= 0.45:
            return "Overstocked"
        return "Healthy"

    def recommend_for_product(
        self,
        product_id: str,
        historical_demand: Sequence[float],
        current_stock: float,
        lead_time: Optional[float] = None,
        review_period: Optional[float] = None,
        safety_stock_factor: Optional[float] = None,
        service_level: Optional[float] = None,
    ) -> InventoryRecommendation:
        """Return a complete recommendation for a single product."""

        values = self._as_float_array(historical_demand)
        forecasted_demand = self._forecast_demand(values)
        demand_std_dev = self._demand_std_dev(values)

        _lead_time = self.config.lead_time if lead_time is None else float(lead_time)
        _review_period = self.config.review_period if review_period is None else float(review_period)
        _safety_stock_factor = (
            self.config.safety_stock_factor
            if safety_stock_factor is None
            else float(safety_stock_factor)
        )
        _service_level = (
            self.config.service_level
            if service_level is None
            else float(service_level)
        )

        safety_stock = self.calculate_safety_stock(
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            safety_stock_factor=_safety_stock_factor,
            service_level=_service_level,
        )
        reorder_point = self.calculate_reorder_point(
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            safety_stock_factor=_safety_stock_factor,
            service_level=_service_level,
        )
        recommended_quantity = self.calculate_recommended_reorder_quantity(
            current_stock=current_stock,
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            review_period=_review_period,
            safety_stock_factor=_safety_stock_factor,
            service_level=_service_level,
        )
        stock_out_risk = self.estimate_stock_out_risk(
            current_stock=current_stock,
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
        )
        excess_inventory_risk = self.detect_excess_inventory(
            current_stock=current_stock,
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            review_period=_review_period,
            safety_stock_factor=_safety_stock_factor,
            service_level=_service_level,
        )
        inventory_status = self.generate_inventory_status(
            current_stock=current_stock,
            forecasted_demand=forecasted_demand,
            demand_std_dev=demand_std_dev,
            lead_time=_lead_time,
            review_period=_review_period,
            safety_stock_factor=_safety_stock_factor,
            service_level=_service_level,
        )

        return InventoryRecommendation(
            product_id=str(product_id),
            current_stock=float(current_stock),
            forecasted_demand=float(forecasted_demand),
            safety_stock=float(safety_stock),
            reorder_point=float(reorder_point),
            recommended_reorder_quantity=float(recommended_quantity),
            stock_out_risk=float(stock_out_risk),
            excess_inventory_risk=float(excess_inventory_risk),
            inventory_status=inventory_status,
            lead_time=float(_lead_time),
            review_period=float(_review_period),
            service_level=float(_service_level),
        )

    def recommend_for_products(
        self,
        product_demands: Dict[str, Sequence[float]],
        current_stock_by_product: Dict[str, float],
    ) -> Dict[str, InventoryRecommendation]:
        """Return recommendations for multiple products."""

        recommendations: Dict[str, InventoryRecommendation] = {}

        for product_id, historical_demand in product_demands.items():
            current_stock = float(current_stock_by_product.get(product_id, 0.0))
            recommendations[product_id] = self.recommend_for_product(
                product_id=product_id,
                historical_demand=historical_demand,
                current_stock=current_stock,
            )

        return recommendations


__all__ = ["InventoryConfig", "InventoryRecommendation", "InventoryManager"]
