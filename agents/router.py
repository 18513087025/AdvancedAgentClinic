
import json
from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt



# TODO: 拆分角色理由，prompt 太长

class RouterAgent(BaseAgent):

    # direct_finalize
    # → 继续走单医生 / 单主控模式
    # initiate_specialist_discussion
    # → 系统切换到多专家模式

    def __init__(
        self,
        sender,
        profile,
        scenario,
        case_store,
        llm_client,
        message_store,
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
        self.case_store = case_store

    def system_prompt(self) -> str:
        runtime_info = {
            "context_title": "TASK OBJECTIVE",
            "context": self.context, 
            "case_info": self.case_store.to_dict()["patient_info"]
            
        }
        return build_system_prompt(self.profile, runtime_info)

    def inference(self, window_id: str) -> ChatMessage: # Retry logic to ensure valid JSON output
        MAX_RETRIES = 2

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()

            if content and content.startswith("{") and content.endswith("}"):
                try:
                    parsed_json = json.loads(content)
                    if isinstance(parsed_json, dict):
                        msg.metadata["output_type"] = "triage"
                        msg.metadata["structured_data"] = parsed_json
                        return msg
                except json.JSONDecodeError:
                    pass

            if retry < MAX_RETRIES:
                self.message_store.add_message(
                    window_id,
                    ChatMessage(
                        sender="system",
                        content=(
                            "Your previous response was invalid.\n"
                            "You must return ONLY a valid JSON object.\n"
                            "Do NOT include any text before or after the JSON.\n"
                            "The JSON must strictly follow the required schema:\n"
                            '{\n'
                            '  "complexity": "low | medium | high",\n'
                            '  "reasoning": "",\n'
                            '  "next_action": "single_doctor | initiate_specialist_discussion"\n'
                            '}'
                        ),
                    ),
                )

        raise ValueError(
            f"[{self.sender}] Failed to produce valid triage JSON after retries.\n"
            f"Last output:\n{content}"
        )