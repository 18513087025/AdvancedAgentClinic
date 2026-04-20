import json
from typing import Any
from core.base_agent import BaseAgent
from configs import profiles
import re


class DoctorAgent(BaseAgent):
    """
        system_prompt 开局信息
    """
    def __init__(self, scenario, llm_client, max_infs: int = 20, img_processing: bool = False) -> None:
        super().__init__("doctor", scenario.examiner_info, llm_client, img_processing=img_processing)
        self.infs = 0
        self.MAX_INFS = max_infs


    def system_prompt(self) -> str:
        return profiles.DOCTOR_PROMPT_TEMPLATE.format(
            MAX_INFS=self.MAX_INFS,
            infs=self.infs,
            image_rule=(
                'You can request test results using the exact format: "REQUEST TEST: [test]". '
                'For example: "REQUEST TEST: Chest_X-Ray".'
                if self.img_processing
                else "You cannot request test results."
            ), 
            presentation = self.presentation, 
        )   
    
    def inference(self, response):
        self.infs += 1
        return super().inference(response)


# ===============================
class PatientAgent(BaseAgent):
    def __init__(self, scenario, llm_client) -> None:
        super().__init__("patient", scenario.patient_info, llm_client)
    
    def system_prompt(self):
        return profiles.PATIENT_PROMPT_TEMPLATE.format(presentation=self.presentation)

# ===============================
class IntakeAgent(BaseAgent):

    def __init__(self, scenario, llm_client, max_infs: int = 20, img_processing: bool = False) -> None:  # img_processing 是否开启llm的图像分析能力
        super().__init__("intake", scenario.examiner_info, llm_client, img_processing=img_processing)
        self.infs = 0
        self.MAX_INFS = max_infs
        self._prompt = f"""
        Now, based on the full dialogue history that you have already collected,
        organize the patient information into a structured clinical summary.

        Return valid JSON only with the following schema:

        [BEGIN JSON]{{
        "patient_profile": {{}},
        "physical_exam": {{}},
        "labs": {{}},
        "imaging": {{}},
        "pathology": {{}}
        }}[END JSON]

        Do not add any explanation.
        Do not hallucinate information.
        
        Below is the dialogue history you have collected:
        {self.agent_hist if self.agent_hist else '[No dialogue collected]'}
        """

    def system_prompt(self):
        return profiles.INTAKE_PROMPT_TEMPLATE.format( # INTAKE_COMPLETE
            MAX_TURNS=self.MAX_INFS,
            turns=self.infs,
            image_rule=(
                "If you believe medical images may be helpful, you should ask the patient whether they can provide prior imaging information such as X-ray, CT, MRI, or ultrasound. "
                "Only if the patient clearly confirms that such images are available, output exactly \"INTAKE_IMAGE\" and nothing else."
                if self.img_processing
                else "You are not allowed to ask the patient about any medical imaging or image-related information."
            ),
            presentation = self.presentation,
        )
    
    def inference(self, response):
        self.infs += 1
        return super().inference(response)
    

    def extract_json_block(self, text: str):
        if text is None:
            return None

        match = re.search(r"\[BEGIN JSON\]\s*(\{[\s\S]*?\})\s*\[END JSON\]", text)

        if not match:
            return None

        json_str = match.group(1)

        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None

        


    def build_structured_patient_info(self, _prompt: str) -> dict[str, Any]:
        result = self.inference(_prompt)
        if result is None:
            return {
                "patient_profile": {},
                "physical_exam": {},
                "labs": {},
                "imaging": {},
                "pathology": {},
            }

        try:
            return self.extract_json_block(result)
        except json.JSONDecodeError:
            return {
                "patient_profile": {},
                "physical_exam": {},
                "labs": {},
                "imaging": {},
                "pathology": {},
            }

    def write_structured_patient_info(self, case_store) -> None:
        structured = self.build_structured_patient_info(self._prompt)
        case_store.set_patient_profile(structured.get("patient_profile", {}))
        case_store.update_physical_exam(structured.get("physical_exam", {}))
        case_store.update_labs(structured.get("labs", {}))
        case_store.update_imaging(structured.get("imaging", {}))
        case_store.update_pathology(structured.get("pathology", {}))
    

    


    