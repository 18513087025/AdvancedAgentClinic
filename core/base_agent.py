
# 共享消息总线 + 聊天窗口视图
# 历史不是 agent 持有的，而是系统持有的；agent 只是读取某个窗口的视图。


# 一、总体架构

# 1. MessageStore
# 全局消息存储。所有消息都写这里。
# 2. ChatWindow
# 一个“聊天窗口”只是消息的一个视图，不一定真的拷贝消息。
# 3. EventBus / Pipe
# agent 之间不直接互调，而是通过事件或管道发送消息。
# 4. Orchestrator
# 主控循环。负责调度哪个 agent 读哪个窗口、往哪个窗口发消息。


from abc import ABC, abstractmethod
from typing import Any
from core.message_store import ChatMessage



class BaseAgent(ABC):
    def __init__(
        self,
        sender: str,
        profile: dict,
        context: Any,
        llm_client,
        message_store,
        img_processing: bool = False,
    ) -> None:
        self.sender = sender
        self.profile = profile
        self.context = context

        self.llm_client = llm_client
        self.img_processing = img_processing
        self.message_store = message_store  # 共享消息存储；orchestrator 负责决定使用哪个窗口，agent 基于该窗口读取上下文并写回回复

    def dialog_input(self, window_id: str) -> str:
        messages = self.message_store.get_window_messages(window_id)
        history = self.message_store.get_window_text(window_id)

        latest = messages[-1].content if messages else "[No message]"
        latest_sender = messages[-1].sender if messages else "[Unknown]"

        return (
            f"[CHAT HISTORY]\n{history}\n\n"
            f"[LATEST MESSAGE]\n{latest_sender}: {latest}\n\n"
            f"Reply as {self.sender}."
        )

    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError

    def build_messages(self, window_id: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.dialog_input(window_id)},
        ]

    def inference(self, window_id: str) -> ChatMessage:
        messages = self.build_messages(window_id)

        try:
            answer = self.llm_client.think(messages, temperature=0)
        except Exception as e:
            raise RuntimeError(f"[{self.sender}] LLM 调用失败: {e}")

        if answer is None:
            raise ValueError(f"[{self.sender}] LLM 返回 None（可能是解析失败或模型异常）")

        if not isinstance(answer, str):
            answer = str(answer)

        msg = self.message_store.append_message(
            window_id=window_id,
            sender=self.sender,
            content=answer,
        )

        return msg

    def get_window_history_text(self, window_id: str) -> str:
        return self.message_store.get_window_text(window_id)