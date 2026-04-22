
from core.base_agent import BaseAgent
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


