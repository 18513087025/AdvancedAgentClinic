import json
import random


class StructuredScenario:
    """
    For MedQA / MedQAExtended / MIMICIV
    """

    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict

        osce = scenario_dict["OSCE_Examination"]

        self.patient_info = osce["Patient_Actor"]
        self.examiner_info = osce["Objective_for_Doctor"]
        self.exam_info = {
            **osce["Physical_Examination_Findings"],
            "tests": osce["Test_Results"],
        }
        self.diagnosis_info = osce["Correct_Diagnosis"]


class NEJMScenario:
    """
    For NEJM / NEJMExtended
    """

    def __init__(self, scenario_dict) -> None:
        self.scenario_dict = scenario_dict

        self.patient_info = scenario_dict["patient_info"]
        self.examiner_info = "What is the most likely diagnosis?"
        self.exam_info = dict(scenario_dict["physical_exams"])
        self.diagnosis_info = [
            ans["text"] for ans in scenario_dict["answers"] if ans["correct"]
        ][0]

        # NEJM specific
        self.image_url = scenario_dict["image_url"]


class ScenarioLoader:
    DATASET_CONFIG = {
        "MedQA": {
            "file_path": "dataset/StructuredData/agentclinic_medqa.jsonl",
            "scenario_cls": StructuredScenario,
        },
        "MedQA_Ext": {
            "file_path": "dataset/StructuredData/agentclinic_medqa_extended.jsonl",
            "scenario_cls": StructuredScenario,
        },
        "MIMICIV": {
            "file_path": "dataset/StructuredData/agentclinic_mimiciv.jsonl",
            "scenario_cls": StructuredScenario,
        },
        "NEJM": {
            "file_path": "dataset/StructuredData/agentclinic_nejm.jsonl",
            "scenario_cls": NEJMScenario,
        },
        "NEJM_Ext": {
            "file_path": "dataset/StructuredData/agentclinic_nejm_extended.jsonl",
            "scenario_cls": NEJMScenario,
        },
    }

    def __init__(self, dataset: str) -> None:
        if dataset not in self.DATASET_CONFIG:
            raise ValueError(f"Unknown dataset: {dataset}")

        config = self.DATASET_CONFIG[dataset]
        self.dataset = dataset
        self.file_path = config["file_path"]
        self.scenario_cls = config["scenario_cls"]

        with open(self.file_path, "r", encoding="utf-8") as f:
            self.scenario_strs = [json.loads(line) for line in f]

        self.scenarios = [self.scenario_cls(item) for item in self.scenario_strs]
        self.num_scenarios = len(self.scenarios)

    def sample_scenario(self):
        return random.choice(self.scenarios)

    def get_scenario(self, id=None):
        if id is None:
            return self.sample_scenario()
        return self.scenarios[id]