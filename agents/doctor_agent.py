import json
import os
from typing import Optional

from dotenv import load_dotenv
from core.llm_client import HelloAgentsLLM

load_dotenv()


class DoctorAgent:
    def __init__(self, scenario, max_infs: int = 20, img_request: bool = False) -> None:
        # number of inference calls to the doctor
        self.infs = 0

        # maximum number of inference calls to the doctor
        self.MAX_INFS = max_infs

        # conversation history between doctor and patient
        self.agent_hist = ""

        # presentation information for doctor
        self.presentation = ""

        # scenario object
        self.scenario = scenario

        # whether image request is allowed
        self.img_request = img_request

        # actual llm client
        self.llm = HelloAgentsLLM(
            os.getenv("DOCTOR_MODEL_ID"),
            os.getenv("DOCTOR_API_KEY"),
            os.getenv("DOCTOR_BASE_URL")
        )

        self.reset()
        self.pipe = None

    def build_dialog_input(self, patient_response: str) -> str:
        """Build the user message passed to the LLM."""
        return (
            f"Here is the history of your dialogue with the patient:\n"
            f"{self.agent_hist if self.agent_hist else '[No previous dialogue]'}\n\n"
            f"The patient just said:\n{patient_response}\n\n"
            f"Please continue the dialogue as the doctor.\n"
            f"Respond in 1-3 sentences only."
        )

    def inference_doctor(
        self,
        patient_response: str,
        scene=None,
        image_requested: bool = False
    ) -> Optional[str]:
        if self.infs >= self.MAX_INFS:
            return "Maximum inferences reached"

        # 默认走纯文本
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.build_dialog_input(patient_response)},
        ]

        # 如果请求图像，并且 scene 里确实有 image_url，就走图文输入
        if image_requested and scene is not None and hasattr(scene, "image_url"):
            try:
                messages = [
                    {"role": "system", "content": self.system_prompt()},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": self.build_dialog_input(patient_response)},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": scene.image_url,
                                },
                            },
                        ],
                    },
                ]
            except Exception:
                pass

        answer = self.llm.think(messages, temperature=0)

        if answer is None:
            return None

        self.agent_hist += f"Patient: {patient_response}\nDoctor: {answer}\n\n"
        self.infs += 1
        return answer

    def system_prompt(self) -> str:
        base = (
            "You are a doctor named Dr. Agent who only responds in the form of dialogue. "
            "You are inspecting a patient and must understand their disease by asking questions. "
            f"You are only allowed to ask {self.MAX_INFS} questions total before making a decision. "
            f"You have asked {self.infs} questions so far. "
            "Your responses must be 1-3 sentences long. "
            'You may request test results using the exact format: "REQUEST TEST: [test]". '
            'For example: "REQUEST TEST: Chest_X-Ray". '
            'Once you are ready to diagnose, output the exact format: "DIAGNOSIS READY: [diagnosis here]". '
        )

        if self.img_request:
            base += 'You may also request medical images using the exact format: "REQUEST IMAGES". '

        presentation = (
            "\n\nBelow is all of the information currently available to you:\n"
            f"{json.dumps(self.presentation, ensure_ascii=False, indent=2) if isinstance(self.presentation, dict) else self.presentation}\n\n"
            "Remember: you must discover the disease by interacting with the patient and requesting appropriate tests."
        )

        return base + presentation

    def reset(self) -> None:
        self.infs = 0
        self.agent_hist = ""
        self.presentation = self.scenario.examiner_information()

    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"