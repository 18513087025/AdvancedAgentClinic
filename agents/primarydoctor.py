import json

from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt


class PrimoryDoctorAgent(BaseAgent):
    def __init__(
        self,
        sender,
        profile,
        scenario,
        llm_client,
        message_store,
        case_store, 
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
            "context": self.context.to_dict(), 
            "case_info": self.case_store.to_dict()["patient_info"]
        }
        return build_system_prompt(self.profile, runtime_info)



    def inference(self, window_id: str) -> ChatMessage:
        self.infs += 1
        MAX_RETRIES = 2

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()

            # ===== 1. REQUEST TEST =====
            if content.upper().startswith("REQUEST TEST:"):
                msg.metadata["done"] = False
                msg.metadata["output_type"] = "request_test"
                msg.metadata["structured_data"] = {
                    "test_request": content.split(":", 1)[1].strip()
                }
                return msg

            # ===== 2. JSON → final diagnosis =====
            if content.startswith("{") and content.endswith("}"):
                try:
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, dict):
                        msg.metadata["done"] = True
                        msg.metadata["output_type"] = "final_diagnosis"
                        msg.metadata["structured_data"] = parsed_json
                        return msg
                except json.JSONDecodeError:
                    pass

            # ===== 3. 非法输出 → retry =====
            if retry < MAX_RETRIES:
                self.message_store.add_message(
                    window_id,
                    ChatMessage(
                        sender="system",
                        content=(
                            "Your previous response was invalid.\n"
                            "You must follow EXACTLY one of the two formats:\n\n"
                            "1. REQUEST TEST: [test]\n"
                            "2. A valid JSON object ONLY\n\n"
                            "Do NOT output any explanation or extra text."
                        ),
                    ),
                )

        # ===== 极端情况 =====
        raise ValueError(
            f"[{self.sender}] Failed to produce valid doctor output after retries.\n"
            f"Last output:\n{content}"
        )