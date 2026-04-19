import os
import json
from typing import Optional
from dotenv import load_dotenv
from Agents import PatientAgent


load_dotenv()



def main():
    # ===== 1. 构造一个假的 scenario（最小可用） =====
    class DummyScenario:
        def patient_information(self):
            return {
                "age": 25,
                "chief_complaint": "abdominal pain",
                "duration": "10 hours",
                "location": "right lower abdomen",
                "associated_symptoms": ["nausea", "loss of appetite"]
            }

    scenario = DummyScenario()

    # ===== 2. 初始化 PatientAgent =====
    patient = PatientAgent(scenario)

    print("=== Start Doctor-Patient Simulation ===\n")

    # ===== 3. 模拟医生问诊 =====
    questions = [
        "Where exactly is your pain located?",
        "When did the pain start?",
        "Do you have any other symptoms?",
    ]

    for q in questions:
        print(f"Doctor: {q}")
        answer = patient.inference_patient(q)
        print(f"Patient: {answer}\n")

    print("=== Conversation End ===")


if __name__ == "__main__":
    main()