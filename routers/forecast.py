from fastapi import APIRouter, Query
from services.forecast_service import (
    forecast_revenue,
    forecast_product_demand,
    get_inventory_analytics
)

router = APIRouter(prefix="/forecast", tags=["Forecast"])

@router.get("/revenue")
def revenue_forecast(periods: int = Query(default=30, ge=7, le=90)):
    return forecast_revenue(periods)

@router.get("/demand")
def demand_forecast(
    product: str = Query(default=None),
    periods: int = Query(default=30, ge=7, le=90)
):
    return forecast_product_demand(product, periods)

@router.get("/inventory-analytics")
def inventory_analytics():
    return get_inventory_analytics()