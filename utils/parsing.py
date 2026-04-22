import json
from typing import Optional


def is_diagnosis_ready(text: Optional[str]) -> bool:
    """Check whether the doctor has provided a final diagnosis."""
    return text is not None and "DIAGNOSIS READY:" in text


def is_request_test(text: Optional[str]) -> bool:
    """Check whether the doctor requests a medical test."""
    return text is not None and "REQUEST TEST:" in text



def is_request_images(text: Optional[str]) -> bool:
    """Check whether the doctor requests medical images."""
    return text is not None and "REQUEST IMAGES" in text


def extract_final_diagnosis(doctor_text: str) -> str:
    """Extract the final diagnosis from doctor output."""
    if doctor_text is None:
        return "UNKNOWN"

    if "DIAGNOSIS READY:" in doctor_text:
        return doctor_text.split("DIAGNOSIS READY:", 1)[1].strip()

    return doctor_text.strip()




def try_parse_json(text: str):
    if not (text.startswith("{") and text.endswith("}")):
        return None
    try:
        data = json.loads(text)

        return data
    except json.JSONDecodeError:
        return None
    return None