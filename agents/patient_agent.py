import json
import os
from typing import Optional

from dotenv import load_dotenv
from core.llm_client import HelloAgentsLLM

load_dotenv()


class PatientAgent:
    def __init__(self, scenario, bias_present=None) -> None:
        self.disease = ""
        self.symptoms = ""
        self.agent_hist = ""
        self.scenario = scenario

        self.llm = HelloAgentsLLM(
            os.getenv("PATIENT_MODEL_ID"),
            os.getenv("PATIENT_API_KEY"),
            os.getenv("PATIENT_BASE_URL")
        )

        self.reset()
        self.pipe = None

    def build_dialog_input(self, question: str) -> str:
        return (
            f"Here is the history of your dialogue with the doctor:\n"
            f"{self.agent_hist if self.agent_hist else '[No previous dialogue]'}\n\n"
            f"The doctor just asked:\n{question}\n\n"
            f"Please continue the dialogue as the patient.\n"
            f"Respond in 1-3 sentences only."
        )

    def inference_patient(self, question: str) -> Optional[str]:
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.build_dialog_input(question)},
        ]

        answer = self.llm.think(messages, temperature=0)

        if answer is None:
            return None

        self.agent_hist += f"Doctor: {question}\nPatient: {answer}\n\n"
        return answer

    def system_prompt(self) -> str:
        base = (
            "You are a patient in a clinic who only responds in dialogue. "
            "A doctor will ask you questions to understand your disease. "
            "Your answer must be 1-3 sentences long. "
            "Do not reveal the diagnosis explicitly. "
            "Only describe symptoms or information that the doctor asks about. "
            "Do not invent facts not contained in your case information."
        )

        symptoms = (
            "\n\nBelow is all of your case information:\n"
            f"{json.dumps(self.symptoms, ensure_ascii=False, indent=2)}\n\n"
            "Remember: you must not explicitly state the diagnosis. "
            "Only answer as the patient in natural dialogue."
        )

        return base + symptoms

    def reset(self) -> None:
        self.agent_hist = ""
        self.symptoms = self.scenario.patient_information()

    def add_hist(self, hist_str) -> None:
        self.agent_hist += hist_str + "\n\n"