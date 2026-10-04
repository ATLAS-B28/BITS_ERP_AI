from prophet import Prophet
from services.db import get_daily_revenue, get_stock_movements, get_inventory_summary
import pandas as pd
import numpy as np
from datetime import datetime

def forecast_revenue(periods: int = 30):
    df = get_daily_revenue()

    if df.empty or len(df) < 2:
        # return dummy projection if no data yet
        dates = pd.date_range(start=datetime.today(), periods=periods)
        return {
            "historical": [],
            "forecast": [
                {
                    "ds": str(d.date()),
                    "yhat": 0,
                    "yhat_lower": 0,
                    "yhat_upper": 0,
                }
                for d in dates
            ],
            "message": "Insufficient data for forecasting — showing placeholder"
        }

    df['ds'] = pd.to_datetime(df['ds'])
    df['y'] = df['y'].astype(float)

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        interval_width=0.80,
    )
    model.fit(df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    historical = df.rename(columns={'ds': 'date', 'y': 'actual'})
    historical['date'] = historical['date'].astype(str)

    result_forecast = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    result_forecast = result_forecast.copy()
    result_forecast['ds'] = result_forecast['ds'].astype(str)
    result_forecast['yhat'] = result_forecast['yhat'].round(2)
    result_forecast['yhat_lower'] = result_forecast['yhat_lower'].round(2)
    result_forecast['yhat_upper'] = result_forecast['yhat_upper'].round(2)

    return {
        "historical": historical.to_dict(orient='records'),
        "forecast": result_forecast.to_dict(orient='records'),
        "message": f"Revenue forecast for next {periods} days"
    }

def forecast_product_demand(product_name: str = None, periods: int = 30):
    df = get_stock_movements()

    if df.empty:
        return {"error": "No stock movement data available"}

    if product_name:
        df = df[df['product_name'].str.lower() == product_name.lower()]

    if df.empty:
        return {"error": f"No data found for product: {product_name}"}

    # aggregate by date
    demand_df = df.groupby('ds')['stock_out'].sum().reset_index()
    demand_df.columns = ['ds', 'y']
    demand_df['ds'] = pd.to_datetime(demand_df['ds'])
    demand_df['y'] = demand_df['y'].astype(float)

    if len(demand_df) < 2:
        return {"error": "Insufficient data for demand forecasting"}

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )
    model.fit(demand_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
    result = result.copy()
    result['ds'] = result['ds'].astype(str)
    result['yhat'] = result['yhat'].clip(lower=0).round(2)
    result['yhat_lower'] = result['yhat_lower'].clip(lower=0).round(2)
    result['yhat_upper'] = result['yhat_upper'].round(2)

    historical = demand_df.copy()
    historical['ds'] = historical['ds'].astype(str)

    return {
        "product": product_name or "All Products",
        "historical": historical.rename(
            columns={'ds': 'date', 'y': 'demand'}
        ).to_dict(orient='records'),
        "forecast": result.to_dict(orient='records'),
        "message": f"Demand forecast for next {periods} days"
    }

def get_inventory_analytics():
    df = get_inventory_summary()

    if df.empty:
        return {"categories": [], "products": [], "total_value": 0}

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    # category breakdown
    category_df = df.groupby('category').agg(
        total_qty=('total_qty', 'sum'),
        total_value=('stock_value', 'sum'),
        product_count=('name', 'count')
    ).reset_index()
    category_df = category_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    return {
        "categories": category_df.to_dict(orient='records'),
        "products": df.to_dict(orient='records'),
        "total_value": float(df['stock_value'].sum()),
        "total_products": len(df),
    }