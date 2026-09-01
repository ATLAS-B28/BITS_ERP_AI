from fastapi import APIRouter
from pydantic import BaseModel
from typing import List
from services.chat_service import chat_with_erp

router = APIRouter(prefix="/chat", tags=["Chat"])

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@router.post("/")
def chat(request: ChatRequest):
    messages = [{"role": m.role, "content": m.content}
                for m in request.messages]
    return chat_with_erp(messages)

@router.get("/suggestions")
def get_suggestions():
    return {
        "suggestions": [
            "What is the current stock value?",
            "Which items are low on stock?",
            "Show me the purchase order summary",
            "What is the sales revenue status?",
            "Which budgets are close to being exceeded?",
            "What are the top products by value?",
        ]
    }