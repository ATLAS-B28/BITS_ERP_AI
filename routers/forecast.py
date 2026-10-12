from fastapi import APIRouter, Query
from services.forecast_service import (
    forecast_revenue,
    forecast_product_demand,
    get_inventory_analytics
)
from services.forecast_agent import (
    forecast_revenue_enhanced,
    forecast_product_demand_enhanced,
    get_inventory_projection,
)

router = APIRouter(prefix="/forecast", tags=["Forecast"])

@router.get("/revenue")
def revenue_forecast(periods: int = Query(default=30, ge=7, le=90)):
    return forecast_revenue(periods)

@router.get("/revenue-enhanced")
def revenue_forecast_enhanced(periods: int = Query(default=30, ge=7, le=90)):
    return forecast_revenue_enhanced(periods)

@router.get("/demand")
def demand_forecast(
    product: str = Query(default=None),
    periods: int = Query(default=30, ge=7, le=90)
):
    return forecast_product_demand(product, periods)

@router.get("/demand-enhanced")
def demand_forecast_enhanced(
    product: str = Query(default=None),
    periods: int = Query(default=30, ge=7, le=90)
):
    return forecast_product_demand_enhanced(product, periods)

@router.get("/inventory-analytics")
def inventory_analytics():
    return get_inventory_analytics()

@router.get("/inventory-projection")
def inventory_projection(periods: int = Query(default=30, ge=7, le=90)):
    return get_inventory_projection(periods)