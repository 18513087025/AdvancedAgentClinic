import json

from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt


class CoordinatorAgent(BaseAgent):
    def __init__(
        self,
        sender,
        profile,
        scenario,
        case_store, 
        llm_client,
        message_store,
        max_discussion_turns: int = 3,
        img_processing: bool = False,
    ) -> None:
        super().__init__(
            sender,
            profile,
            scenario.examiner_info,
            llm_client,
            message_store,
            img_processing,
        )
        self.max_discussion_turns = max_discussion_turns
        self.infs = 0
        self.case_store = case_store

    def system_prompt(self) -> str:
        runtime_info = {
            "max_turns": self.max_discussion_turns,
            "turns": self.infs,
            "context_title": "TASK OBJECTIVE",
            "context": self.context, 
            "case_info": self.case_store.to_dict()["patient_info"]
        }
        return build_system_prompt(self.profile, runtime_info)

    def inference(self, window_id: str) -> ChatMessage:
        self.infs += 1
        MAX_RETRIES = 2

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()

            # ===== 1. 非 JSON → 正常协调对话（优先）=====
            if not (content.startswith("{") and content.endswith("}")):
                msg.metadata["done"] = False
                msg.metadata["output_type"] = "dialogue"
                msg.metadata["structured_data"] = None
                return msg

            # ===== 2. 是 JSON → 尝试解析 final report =====
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    msg.metadata["done"] = True
                    msg.metadata["output_type"] = "final_report"
                    msg.metadata["structured_data"] = parsed_json
                    return msg
            except json.JSONDecodeError:
                pass

            # ===== 3. JSON 坏了 → retry =====
            if retry < MAX_RETRIES:
                self.message_store.add_message(
                    window_id,
                    ChatMessage(
                        sender="system",
                        content=(
                            "Your previous output was not valid JSON.\n"
                            "You must return ONLY a valid JSON object.\n"
                            "Do NOT include any text before or after the JSON."
                        ),
                    ),
                )

        # ===== 4. 极端情况 =====
        raise ValueError(
            f"[{self.sender}] Failed to produce valid final report JSON after retries.\n"
            f"Last output:\n{content}"
        )