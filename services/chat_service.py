from groq import Groq
from services.db import (
    get_inventory_summary, get_po_summary,
    get_sales_summary, get_budget_summary, get_low_stock
)
from dotenv import load_dotenv
import os
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

def build_erp_context():
    try:
        inventory = get_inventory_summary()
        po = get_po_summary()
        sales = get_sales_summary()
        budgets = get_budget_summary()
        low_stock = get_low_stock()

        context = f"""
You are an ERP assistant for BITS ERP — a retail and logistics management system.
You have access to real-time data from the system. Answer questions clearly and concisely.

=== INVENTORY SUMMARY ===
Total Products: {len(inventory)}
Total Stock Value: ₹{inventory['stock_value'].sum():,.2f}
Top products by value:
{inventory.head(5)[['name','category','total_qty','stock_value']].to_string(index=False)}

=== LOW STOCK ALERTS ===
{f"Items below reorder level: {len(low_stock)}" if not low_stock.empty else "All stock levels healthy"}
{low_stock[['name','quantity','reorder_level','deficit']].to_string(index=False) if not low_stock.empty else ""}

=== PURCHASE ORDERS ===
{po[['status','count','total']].to_string(index=False) if not po.empty else "No purchase orders"}

=== SALES ORDERS ===
{sales[['status','count','total']].to_string(index=False) if not sales.empty else "No sales orders"}

=== ACTIVE BUDGETS ===
{budgets[['name','module','allocated_amount','spent_amount','remaining']].to_string(index=False) if not budgets.empty else "No active budgets"}

Answer questions about this ERP data. Be specific with numbers when available.
If asked about something not in the data, say you don't have that information.
Keep responses concise — 2-4 sentences maximum unless a list is needed.
"""
        return context
    except Exception as e:
        return f"ERP Assistant ready. (Context fetch error: {str(e)})"

def chat_with_erp(messages: list, refresh_context: bool = False):
    context = build_erp_context()

    system_message = {
        "role": "system",
        "content": context
    }

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[system_message] + messages,
        temperature=0.3,
        max_tokens=500,
    )

    return {
        "reply": response.choices[0].message.content,
        "model": response.model,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
    }