import os
import json
from typing import Optional
from dotenv import load_dotenv
from Agents import DoctorAgent


load_dotenv()


# ===== 你自己的 HelloAgentsLLM / DoctorAgent 需要已定义 =====


class DummyScenario:
    """最小可用的假场景，只提供 DoctorAgent 需要的方法。"""

    def examiner_information(self):
        return {
            "Objective_for_Doctor": "Assess the patient with abdominal pain and identify the most likely diagnosis."
        }


def main():
    # 1. 构造一个假的 scenario
    scenario = DummyScenario()

    # 2. 初始化 DoctorAgent
    doctor = DoctorAgent(scenario=scenario, max_infs=5, img_request=False)

    print("=== DoctorAgent Test Start ===\n")

    # 3. 查看系统提示词
    print("----- system_prompt -----")
    print(doctor.system_prompt())
    print()

    # 4. 模拟病人的几轮回答
    patient_responses = [
        "Hello doctor, I have abdominal pain.",
        "It started about 10 hours ago.",
        "The pain is mostly on the lower right side now.",
        "I also feel nauseous.",
    ]

    for i, patient_msg in enumerate(patient_responses, start=1):
        print(f"Round {i}")
        print(f"Patient: {patient_msg}")

        doctor_reply = doctor.inference_doctor(patient_msg)
        print(f"Doctor: {doctor_reply}")
        print(f"infs: {doctor.infs}")
        print("----- current history -----")
        print(doctor.agent_hist)
        print()

    print("=== DoctorAgent Test End ===")


if __name__ == "__main__":
    main()


