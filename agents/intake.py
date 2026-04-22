
import json
from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt


class IntakeAgent(BaseAgent):
    def __init__(
        self,
        sender,
        profile,
        scenario,
        llm_client,
        message_store,
        max_intake_turns: int = 5,
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
        self.max_intake_turns = max_intake_turns
        self.infs = 0

    def system_prompt(self) -> str:
        runtime_info = {
            "max_turns": self.max_intake_turns,
            "turns": self.infs,
            "image_rule": (
                "If you believe medical images may be helpful, you should ask the patient whether they can provide prior imaging information such as X-ray, CT, MRI, or ultrasound. "
                "Only if the patient clearly confirms that such images are available, output exactly \"INTAKE_IMAGE\" and nothing else."
                if self.img_processing
                else "You are not allowed to ask the patient about any medical imaging or image-related information."
            ),
            "context_title": "TASK OBJECTIVE",
            "context": self.context,
        }
        return build_system_prompt(self.profile, runtime_info)

    
    def inference(self, window_id: str) -> ChatMessage:
        self.infs += 1
        MAX_RETRIES = 2

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()

            # ===== 非 JSON → 正常对话（优先处理）=====
            if not (content.startswith("{") and content.endswith("}")):
                msg.metadata["done"] = False
                msg.metadata["output_type"] = "dialogue"
                msg.metadata["structured_data"] = None
                return msg

            # ===== 是 JSON → 尝试解析 =====
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    msg.metadata["done"] = True
                    msg.metadata["output_type"] = "structured_intake"
                    msg.metadata["structured_data"] = parsed_json
                    return msg
            except json.JSONDecodeError:
                pass

            # ===== JSON坏了 → retry =====
            if retry < MAX_RETRIES:
                self.message_store.add_message(
                    window_id,
                    ChatMessage(
                        sender="system",
                        content=(
                            "Your previous response was invalid JSON.\n"
                            "You must return ONLY a valid JSON object.\n"
                            "Do NOT include any text before or after the JSON.\n"
                            "The JSON must follow the required schema."
                        ),
                    ),
                )

        raise ValueError(
            f"[{self.sender}] Failed to produce valid intake JSON after retries.\n"
            f"Last output:\n{content}"
        )