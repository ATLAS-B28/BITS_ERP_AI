from sqlalchemy import create_engine, text
from dotenv import load_dotenv
import pandas as pd
import os

load_dotenv()

engine = create_engine(os.getenv("DATABASE_URL"))

def get_daily_revenue():
    query = text("""
        SELECT 
            DATE(created_at) as ds,
            COALESCE(SUM(total_amount), 0) as y
        FROM sales_orders
        WHERE status = 'DELIVERED'
        GROUP BY DATE(created_at)
        ORDER BY ds
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_stock_movements():
    query = text("""
        SELECT 
            DATE(moved_at) as ds,
            p.name as product_name,
            p.category,
            SUM(CASE WHEN change_qty > 0 THEN change_qty ELSE 0 END) as stock_in,
            SUM(CASE WHEN change_qty < 0 THEN ABS(change_qty) ELSE 0 END) as stock_out
        FROM stock_movements sm
        JOIN products p ON sm.product_id = p.id
        GROUP BY DATE(moved_at), p.name, p.category
        ORDER BY ds
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_inventory_summary():
    query = text("""
        SELECT 
            p.name, p.category, p.unit_price,
            COALESCE(SUM(i.quantity), 0) as total_qty,
            COALESCE(SUM(i.quantity * p.unit_price), 0) as stock_value,
            MIN(i.reorder_level) as reorder_level
        FROM products p
        LEFT JOIN inventory i ON p.id = i.product_id
        WHERE p.active = true
        GROUP BY p.id, p.name, p.category, p.unit_price
        ORDER BY stock_value DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_po_summary():
    query = text("""
        SELECT 
            status,
            COUNT(*) as count,
            COALESCE(SUM(total_amount), 0) as total
        FROM purchase_orders
        GROUP BY status
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_sales_summary():
    query = text("""
        SELECT 
            status,
            COUNT(*) as count,
            COALESCE(SUM(total_amount), 0) as total
        FROM sales_orders
        GROUP BY status
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_budget_summary():
    query = text("""
        SELECT 
            name, module, allocated_amount,
            spent_amount, status,
            (allocated_amount - spent_amount) as remaining
        FROM budgets
        WHERE status = 'ACTIVE'
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df

def get_low_stock():
    query = text("""
        SELECT 
            p.name, p.category,
            i.quantity, i.reorder_level,
            (i.reorder_level - i.quantity) as deficit
        FROM inventory i
        JOIN products p ON i.product_id = p.id
        WHERE i.quantity <= i.reorder_level
        ORDER BY deficit DESC
    """)
    with engine.connect() as conn:
        df = pd.read_sql(query, conn)
    return df