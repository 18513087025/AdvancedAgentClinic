import json

from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt


class SpecialistAgent(BaseAgent):
    def __init__(
        self,
        sender,
        profile,
        scenario,
        case_store,
        llm_client,
        message_store,
        specialty: str,
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
        self.specialty = specialty
        self.max_discussion_turns = max_discussion_turns
        self.infs = 0
        self.case_store = case_store

    def system_prompt(self) -> str:
        # is_last_turn = (self.infs + 1) >= self.max_discussion_turns
        runtime_info = {
            "max_turns": self.max_discussion_turns,
            "turns": self.infs,
            "context_title": "TASK OBJECTIVE",
            "context": self.context,
            "case_info": self.case_store.to_dict()["patient_info"],
            # "is_last_turn": is_last_turn,
        }
        return build_system_prompt(self.profile, runtime_info)

    def inference(self, window_id: str) -> ChatMessage:
        MAX_RETRIES = 2

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()

            # ===== 如果不是 JSON → 当作讨论 =====
            if not (content.startswith("{") and content.endswith("}")):
                msg.metadata["output_type"] = "discussion"
                msg.metadata["done"] = False
                return msg

            # ===== 是 JSON → 尝试解析 final opinion =====
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    msg.metadata["output_type"] = "final_opinion"
                    msg.metadata["structured_data"] = parsed_json
                    msg.metadata["done"] = True
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
                            "Your previous response was invalid.\n"
                            "You must return ONLY a valid JSON object.\n"
                            "Do NOT include any text before or after the JSON.\n"
                            "The JSON must strictly follow the required schema:\n"
                            "{\n"
                            f'  "specialist": "{self.specialty}",\n'
                            '  "opinion": "",\n'
                            '  "supporting_evidence": [\n'
                            "    {\n"
                            '      "source": "patient_profile | physical_exam | labs | imaging | pathology | specialist_discussion",\n'
                            '      "content": "",\n'
                            '      "confidence": "low | medium | high"\n'
                            "    }\n"
                            "  ]\n"
                            "}"
                        ),
                    ),
                )

        raise ValueError(
            f"[{self.sender}] Failed to produce valid specialist JSON after retries.\n"
            f"Last output:\n{content}"
        )