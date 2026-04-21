from dataclasses import dataclass, field
from typing import Any
import time
import uuid

# 单条消息
@dataclass
class ChatMessage:
    message_id: str
    window_id: str
    sender: str
    content: str
    role: str = "assistant"   # system / user / assistant / tool # TODO: 先占位保留
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)  # metadata 用于存储一些额外信息，比如是否包含图片、图片URL等


# 聊天频道: intake_patient_window / specialist_board_window / measurement_window
@dataclass
class ChatWindow: 
    window_id: str
    title: str
    participants: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

# 全局消息仓库
class MessageStore:
    def __init__(self) -> None:
        self.messages: list[ChatMessage] = []
        self.windows: dict[str, ChatWindow] = {}

    def create_window(self, window_id: str, title: str, participants: list[str]) -> None:
        self.windows[window_id] = ChatWindow(
            window_id=window_id,
            title=title,
            participants=participants,
        )

    def append_message(
        self,
        window_id: str,
        sender: str,
        content: str,
        role: str = "assistant",
        metadata: dict | None = None,
    ) -> ChatMessage:
        msg = ChatMessage(
            message_id=str(uuid.uuid4()), # 唯一标识
            window_id=window_id,
            sender=sender,
            content=content,
            role=role,
            metadata=metadata or {},
        )
        self.messages.append(msg)
        return msg

    def get_window_messages(self, window_id: str) -> list[ChatMessage]:
        return [m for m in self.messages if m.window_id == window_id]

    def get_window_text(self, window_id: str) -> str:
        msgs = self.get_window_messages(window_id)
        if not msgs:
            return "[No previous messages]"
        return "\n\n".join(f"{m.sender}: {m.content}" for m in msgs)
