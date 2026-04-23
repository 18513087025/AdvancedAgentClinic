import json

from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt


class MeasurementAgent(BaseAgent):
    def __init__(
        self,
        sender,
        profile,
        scenario,
        llm_client,
        message_store,
        img_processing: bool = False,
    ) -> None:
        super().__init__(
            sender,
            profile,
            scenario.exam_info,
            llm_client,
            message_store,
            img_processing,
        )

    def system_prompt(self) -> str:
        runtime_info = {
            "context_title": "AVAILABLE MEASUREMENT DATA",
            "contex": self.context, 
      
        }
        return build_system_prompt(self.profile, runtime_info)
  

    def inference(self, window_id: str) -> ChatMessage:

        MAX_RETRIES = 2

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()

            if content.startswith("{") and content.endswith("}"):
                try:
                    parsed = json.loads(content)
                    if isinstance(parsed, dict):
                        msg.metadata["done"] = True
                        msg.metadata["output_type"] = "measurement_result"
                        msg.metadata["structured_data"] = parsed
                        return msg
                except json.JSONDecodeError:
                    pass

            if retry < MAX_RETRIES:
                self.message_store.append_message(
                    window_id=window_id,
                    sender="SYSTEM",
                    
                    content=(
                        "Your previous response was invalid.\n"
                        "You must return ONLY a valid JSON object.\n"
                        "Do NOT include any text before or after the JSON.\n"
                        "The JSON must strictly follow the required schema."
                    ),
                    
                )

        raise ValueError(
            f"[{self.sender}] Failed to produce valid measurement JSON after retries.\n"
            f"Last output:\n{content}"
        )