import json

from core.base_agent import BaseAgent
from core.message_store import ChatMessage
from configs.prompt_builder import build_system_prompt


class PatientAgent(BaseAgent):
    def __init__(
        self,
        sender,
        profile,
        scenario,
        llm_client,
        message_store,
        img_processing: bool = False,
    ):
        super().__init__(
            sender,
            profile,
            scenario.examiner_info,
            llm_client,
            message_store,
            img_processing,
        )

    def system_prompt(self) -> str:
        runtime_info = {
            "context_title": "CASE SCRIPT",
            "context": self.context,
        }
        return build_system_prompt(self.profile, runtime_info)


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
        msg = super().inference(window_id)

        done = "INTAKE DONE" in msg.content
        msg.metadata["intake_done"] = done

        return msg

    def finalize_intake(self, case_store, window_id: str) -> None:
        dialogue_history = self.message_store.get_window_text(window_id)

        base_prompt = f"""
            Now, based on the full dialogue history, organize the patient information.
            Return a valid JSON object with the following schema:

            {{
                "patient_profile": {{}},
                "physical_exam": {{}},
                "labs": {{}},
                "imaging": {{}},
                "pathology": {{}}
            }}

            Strict requirements:
            - Output ONLY JSON
            - No explanation
            - No markdown
            - No extra text

            Dialogue history:
            {dialogue_history}
            """

        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": base_prompt},
        ]

        max_retries = 2
        answer = None

        for retry_id in range(max_retries + 1):
            try:
                answer = self.llm_client.think(messages, temperature=0)
            except Exception as e:
                if retry_id == max_retries:
                    raise RuntimeError(f"[{self.sender}] Intake structured extraction failed: {e}")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response failed. "
                        "Please strictly follow the output format and return VALID JSON ONLY."
                    ),
                })
                continue

            if answer is None:
                if retry_id == max_retries:
                    raise ValueError(f"[{self.sender}] Intake structured extraction returned None")
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was empty. "
                        "Please strictly follow the output format and return VALID JSON ONLY."
                    ),
                })
                continue

            if not isinstance(answer, str):
                answer = str(answer)

            try:
                structured = json.loads(answer)
                break
            except json.JSONDecodeError:
                if retry_id == max_retries:
                    raise ValueError(
                        f"[{self.sender}] Intake structured extraction failed: invalid JSON\n{answer}"
                    )
                messages.append({
                    "role": "assistant",
                    "content": answer,
                })
                messages.append({
                    "role": "user",
                    "content": (
                        "Your previous response was not valid JSON. "
                        "Please strictly follow the output format and return ONLY a valid JSON object."
                    ),
                })
        else:
            raise ValueError(f"[{self.sender}] Intake structured extraction failed after retries")

        mapping = {
            "patient_profile": case_store.set_patient_profile,
            "physical_exam": case_store.update_physical_exam,
            "labs": case_store.update_labs,
            "imaging": case_store.update_imaging,
            "pathology": case_store.update_pathology,
        }

        for key, func in mapping.items():
            value = structured.get(key)
            if not isinstance(value, dict):
                value = {}
            func(value)

        case_store.save_json("./dataset/case_store.jsonl")
        print("Intake complete signal detected. Ending consultation.")


class MasterAgent(BaseAgent):

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
        max_discuss_turns: int = 5,
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
        self.max_discuss_turns = max_discuss_turns
        self.case_store = case_store
        self.infs = 0

    def _build_case_info(self) -> dict:
        data = self.case_store.to_dict()
        return {
            "case_id": data.get("case_id"),
            "patient_info": data.get("patient_info", {}),
            "final_report": data.get("final_report", ""),
        }

    def system_prompt(self) -> str:
        runtime_info = {
            "max_turns": self.max_discuss_turns,
            "turns": self.infs,
            "context_title": "TASK OBJECTIVE",
            "context": self.context,
            "case_title": "CASE",
            "case_info": self._build_case_info(),
            "image_rule": (
                "If enabled, you may approve image-related measurements requested by specialists."
                if self.img_processing
                else "Do not approve image-related measurements."
            ),
        }
        return build_system_prompt(self.profile, runtime_info)

    def inference(self, window_id: str) -> ChatMessage:
        self.infs += 1
        return super().inference(window_id)