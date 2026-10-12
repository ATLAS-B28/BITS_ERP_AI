from prophet import Prophet
import pandas as pd
import numpy as np
from datetime import datetime

from services.db import get_daily_revenue, get_stock_movements, get_inventory_summary


def normalize_series(df, date_col='ds', value_col='y'):
    if df is None or df.empty:
        return pd.DataFrame(columns=[date_col, value_col])

    normalized = df[[date_col, value_col]].copy()
    normalized[date_col] = pd.to_datetime(normalized[date_col])
    normalized[value_col] = pd.to_numeric(normalized[value_col], errors='coerce').fillna(0)
    normalized = normalized.sort_values(date_col).drop_duplicates(subset=date_col)

    if normalized.empty:
        return normalized

    full_range = pd.date_range(start=normalized[date_col].min(), end=normalized[date_col].max(), freq='D')
    full_df = pd.DataFrame({date_col: full_range})
    normalized = full_df.merge(normalized, on=date_col, how='left').fillna({value_col: 0})
    return normalized


def detect_trend(values):
    if len(values) < 2:
        return 'insufficient_data'

    start = float(values.iloc[0])
    end = float(values.iloc[-1])
    delta = (end - start) / max(abs(start), 1)

    if delta > 0.1:
        return 'upward'
    if delta < -0.1:
        return 'downward'
    return 'flat'


def detect_seasonality(values):
    if len(values) < 14:
        return 'limited'

    recent = values.tail(7).mean()
    previous = values.iloc[-14:-7].mean()
    if recent > 0 and abs(recent - previous) > 0.15 * max(previous, 1):
        return 'weekly_pattern'
    return 'limited'


def detect_outliers(values):
    if len(values) < 3:
        return []

    mean_value = float(values.mean())
    std_value = float(values.std(ddof=0))
    if std_value == 0:
        return []

    z_scores = (values - mean_value) / std_value
    return values[z_scores.abs() > 2].index.tolist()


def calculate_confidence(values):
    if len(values) < 10:
        return 'low'
    if len(values) < 30:
        return 'medium'
    return 'high'


def build_recommendation(trend, confidence, anomalies):
    if confidence == 'low':
        return 'Forecast confidence is low because the history is too short or unstable.'
    if anomalies:
        return 'The forecast includes unusual recent activity; review the latest operational changes before acting.'
    if trend == 'upward':
        return 'Trend is upward; keep a close watch on demand and inventory planning.'
    if trend == 'downward':
        return 'Trend is downward; consider reducing stock and tightening procurement.'
    return 'Trend is stable; continue monitoring for sudden pattern changes.'


def _empty_placeholder(periods, label):
    dates = pd.date_range(start=datetime.today(), periods=periods)
    return {
        'historical': [],
        'forecast': [
            {
                'ds': str(d.date()),
                'yhat': 0,
                'yhat_lower': 0,
                'yhat_upper': 0,
            }
            for d in dates
        ],
        'confidence': 'low',
        'trend': 'insufficient_data',
        'seasonality': 'limited',
        'anomalies': [],
        'recommendation': 'Not enough history for a reliable forecast yet.',
        'message': f'Insufficient {label} data for forecasting — showing placeholder output'
    }


def forecast_revenue_enhanced(periods: int = 30):
    df = normalize_series(get_daily_revenue())

    if df.empty or len(df) < 2:
        return _empty_placeholder(periods, 'revenue')

    trend = detect_trend(df['y'])
    seasonality = detect_seasonality(df['y'])
    anomalies = detect_outliers(df['y'])

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
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    result['ds'] = result['ds'].astype(str)
    result['yhat'] = result['yhat'].round(2)
    result['yhat_lower'] = result['yhat_lower'].round(2)
    result['yhat_upper'] = result['yhat_upper'].round(2)

    historical = df.rename(columns={'ds': 'date', 'y': 'actual'}).copy()
    historical['date'] = historical['date'].astype(str)

    confidence = calculate_confidence(df['y'])
    recommendation = build_recommendation(trend, confidence, anomalies)

    return {
        'historical': historical.to_dict(orient='records'),
        'forecast': result.to_dict(orient='records'),
        'confidence': confidence,
        'trend': trend,
        'seasonality': seasonality,
        'anomalies': [
            str(df.iloc[idx]['ds'].date())
            for idx in anomalies
        ],
        'recommendation': recommendation,
        'message': f'Revenue forecast for next {periods} days with quality checks applied'
    }


def forecast_product_demand_enhanced(product_name: str = None, periods: int = 30):
    df = get_stock_movements()

    if df.empty:
        return {'error': 'No stock movement data available'}

    if product_name:
        df = df[df['product_name'].str.lower() == product_name.lower()]

    if df.empty:
        return {'error': f'No data found for product: {product_name}'}

    demand_df = df.groupby('ds')['stock_out'].sum().reset_index()
    demand_df.columns = ['ds', 'y']
    demand_df = normalize_series(demand_df)

    if demand_df.empty or len(demand_df) < 2:
        return {'error': 'Insufficient data for demand forecasting'}

    trend = detect_trend(demand_df['y'])
    seasonality = detect_seasonality(demand_df['y'])
    anomalies = detect_outliers(demand_df['y'])

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=True,
        daily_seasonality=False,
        interval_width=0.80,
    )
    model.fit(demand_df)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods).copy()
    result['ds'] = result['ds'].astype(str)
    result['yhat'] = result['yhat'].clip(lower=0).round(2)
    result['yhat_lower'] = result['yhat_lower'].clip(lower=0).round(2)
    result['yhat_upper'] = result['yhat_upper'].round(2)

    historical = demand_df.rename(columns={'ds': 'date', 'y': 'demand'}).copy()
    historical['date'] = historical['date'].astype(str)

    confidence = calculate_confidence(demand_df['y'])
    recommendation = build_recommendation(trend, confidence, anomalies)

    return {
        'product': product_name or 'All Products',
        'historical': historical.to_dict(orient='records'),
        'forecast': result.to_dict(orient='records'),
        'confidence': confidence,
        'trend': trend,
        'seasonality': seasonality,
        'anomalies': [
            str(demand_df.iloc[idx]['ds'].date())
            for idx in anomalies
        ],
        'recommendation': recommendation,
        'message': f'Demand forecast for next {periods} days with quality checks applied'
    }


def get_inventory_projection(periods: int = 30):
    df = get_inventory_summary()

    if df.empty:
        return {'items': [], 'message': 'No inventory data available'}

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    items = []
    for _, row in df.iterrows():
        qty = float(row.get('total_qty', 0) or 0)
        reorder = float(row.get('reorder_level', 0) or 0)
        stock_value = float(row.get('stock_value', 0) or 0)
        projected_qty = max(0, qty - (qty * 0.15 * (periods / 30)))
        risk = 'high' if projected_qty <= reorder else ('medium' if projected_qty <= reorder * 1.5 else 'low')

        items.append({
            'name': row.get('name'),
            'category': row.get('category'),
            'current_qty': qty,
            'projected_qty': round(projected_qty, 2),
            'reorder_level': reorder,
            'stock_value': round(stock_value, 2),
            'risk': risk,
        })

    return {
        'items': items,
        'total_value': round(float(df['stock_value'].sum()), 2),
        'message': f'Inventory projection for next {periods} days'
    }
