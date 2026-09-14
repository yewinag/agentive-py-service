from fastapi import APIRouter, Depends

from app.chats.schemas import ChatRequest, ChatResponse
from app.chats.service import ChatService, get_chat_service

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
async def create_chat_message(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    reply = await chat_service.get_reply(payload.message)
    return ChatResponse(reply=reply)
