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
        runtime_info = {
            "max_turns": self.max_discussion_turns,
            "turns": self.infs,
            "context_title": "TASK OBJECTIVE",
            "context": self.context,
            "case_info": self.case_store.to_dict()["patient_info"],
        }
        return build_system_prompt(self.profile, runtime_info)

    def inference(self, window_id: str) -> ChatMessage:
        MAX_RETRIES = 2
        last_content = ""

        for retry in range(MAX_RETRIES + 1):
            msg = super().inference(window_id)
            content = (msg.content or "").strip()
            last_content = content

            # ===== 1. REQUEST TEST =====
            if "REQUEST TEST" in content.upper():
                test_name = content.split(":", 1)[1].strip()

                msg.metadata["output_type"] = "request_test"
                msg.metadata["structured_data"] = {
                    "specialist": self.specialty,
                    "test_request": test_name,
                }
                msg.metadata["done"] = False
                return msg

            # ===== 2. 普通文本讨论 =====
            if not (content.startswith("{") and content.endswith("}")):
                msg.metadata["output_type"] = "discussion"
                msg.metadata["done"] = False
                return msg

            # ===== 3. JSON -> final opinion =====
            try:
                parsed_json = json.loads(content)
                if isinstance(parsed_json, dict):
                    msg.metadata["output_type"] = "final_opinion"
                    msg.metadata["structured_data"] = parsed_json
                    msg.metadata["done"] = True
                    return msg
            except json.JSONDecodeError:
                pass

            # ===== 4. JSON 格式坏了，重试 =====
            if retry < MAX_RETRIES:
                self.message_store.add_message(
                    window_id,
                    ChatMessage(
                        sender="SYSTEM",
                        content=(
                            "Your previous response was invalid.\n"
                            "You must output one of the following only:\n"
                            "1. A normal discussion sentence or paragraph; OR\n"
                            "2. A test request in the exact format: REQUEST TEST: [test_name]; OR\n"
                            "3. A valid JSON object for your final opinion.\n\n"
                            "If you output JSON, it must strictly follow this schema:\n"
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
                            "}\n"
                            "Do not include any extra text before or after the JSON."
                        ),
                    ),
                )

        raise ValueError(
            f"[{self.sender}] Failed to produce valid specialist output after retries.\n"
            f"Last output:\n{last_content}"
        )