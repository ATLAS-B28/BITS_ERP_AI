from groq import Groq
import groq as _groq
from services.db import (
    get_inventory_summary, get_po_summary,
    get_sales_summary, get_budget_summary, get_low_stock
)
from dotenv import load_dotenv
import os
import json

load_dotenv()

# Initialize Groq client from env
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
DEFAULT_GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3.8-27b")


def build_erp_context():
    try:
        inventory = get_inventory_summary()
        po = get_po_summary()
        sales = get_sales_summary()
        budgets = get_budget_summary()
        low_stock = get_low_stock()

        context = f"""
You are an ERP assistant for BITS ERP — a retail and logistics system.
Answer questions clearly and concisely based on the data below.

=== INVENTORY ===
Total Products: {len(inventory)}
Total Stock Value: ₹{float(inventory['stock_value'].sum()):,.2f}
{inventory.head(5)[['name','total_qty','stock_value']].to_string(index=False) if not inventory.empty else "No products"}

=== LOW STOCK ===
{f"{len(low_stock)} items below reorder level" if not low_stock.empty else "All stock healthy"}

=== PURCHASE ORDERS ===
{po.to_string(index=False) if not po.empty else "No purchase orders"}

=== SALES ORDERS ===
{sales.to_string(index=False) if not sales.empty else "No sales orders"}

=== BUDGETS ===
{budgets.to_string(index=False) if not budgets.empty else "No active budgets"}
"""
        return context.strip()

    except Exception as e:
        print(f"Context build error: {e}")
        return "You are an ERP assistant for BITS ERP. The database context could not be loaded right now. Answer general ERP questions."


def chat_with_erp(messages: list):
    context = build_erp_context()
    print(f"Context length: {len(context)} chars")

    system_message = {
        "role": "system",
        "content": context
    }

    model_name = os.getenv("GROQ_MODEL", DEFAULT_GROQ_MODEL)

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[system_message] + messages,
            temperature=0.3,
            max_tokens=500,
        )
        print(f"Response preview: {str(response.choices[0].message.content)[:120]}")
        return {
            "reply": response.choices[0].message.content,
            "model": response.model,
            "usage": {
                "prompt_tokens": getattr(response.usage, 'prompt_tokens', None),
                "completion_tokens": getattr(response.usage, 'completion_tokens', None),
            }
        }
    except _groq.NotFoundError as e:
        print(f"Model not found: {e}")
        return {"error": "model_not_found", "message": str(e)}
    except Exception as e:
        print(f"Groq error: {e}")
        return {"error": "api_error", "message": str(e)}