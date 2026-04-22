import json
from copy import deepcopy
from typing import Any

# TODO: 谁能写什么、measurement 怎么规范化、specialist opinion schema 怎么统一
    # patient_profile / exam / labs 的字段规范：否则 intake 抽取出来的 key 会越来越散
# TODO: 是否需要给intake一份必须收集的表格，让他按照表格来收集病人信息，并且追问对诊断可能有用的信息


class CaseStore:
    """
    整个病例在系统中的共享状态仓库
        PatientAgent 负责“说”
        IntakeAgent 负责“问 + 抽取 + 写入 CaseStore”: 模拟《建档阶段》
    """

    def __init__(self, case_id: str):
        self.case_id = case_id
        self.data = {
            "case_id": case_id,

            # intake / measurement 持续补充后的病例主体
            "patient_info": {
                "patient_profile": {},
                "physical_exam": {},
                "labs": {},
                "imaging": {},
                "pathology": {},
            },

            # master 的路径判断 / 分诊结论
            "triage": {},

            # 专家意见：不是过程，而是每个专家最终给出的摘要意见
            "specialist_opinions": {},

            # 已完成的检查项目及结果
            "measurements": {},   # TODO: 存储结构优化 priority:low

            # 最终总结
            "final_report": {},
        }

    def _deep_merge(self, base: dict[str, Any], patch: dict[str, Any]) -> None:
        for key, value in patch.items():
            if (
                key in base
                and isinstance(base[key], dict)
                and isinstance(value, dict)
            ):
                self._deep_merge(base[key], value)
            else:
                base[key] = deepcopy(value)

    def to_dict(self) -> dict[str, Any]:
        return deepcopy(self.data)

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

    # ===== patient_info =====

    def set_patient_profile(self, patient_profile: dict[str, Any]) -> None:
        self.data["patient_info"]["patient_profile"] = deepcopy(patient_profile)

    def update_physical_exam(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["physical_exam"], patch)

    def update_labs(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["labs"], patch)

    def update_imaging(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["imaging"], patch)

    def update_pathology(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"]["pathology"], patch)

    def update_patient_info(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["patient_info"], patch)

    # ===== triage =====


    def update_triage(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["triage"], patch)

    # ===== specialist_opinions =====


    def update_specialist_opinion(self, specialty: str, patch: dict[str, Any]) -> None:
        if specialty not in self.data["specialist_opinions"]:
            self.data["specialist_opinions"][specialty] = {}
        self._deep_merge(self.data["specialist_opinions"][specialty], patch)


    # ===== measurements =====
    # 这里只存已经完成的检查项目和结果，不存请求过程


    def update_measurement(self, name: str, patch: dict[str, Any]) -> None:
        if name not in self.data["measurements"] or not isinstance(self.data["measurements"][name], dict):
            self.data["measurements"][name] = {}
        self._deep_merge(self.data["measurements"][name], patch)

    def update_patient_info_from_measurement(
        self,
        category: str,
        patch: dict[str, Any],
    ) -> None:
        mapping = {
            "physical_exam": self.update_physical_exam,
            "labs": self.update_labs,
            "imaging": self.update_imaging,
            "pathology": self.update_pathology,
        }
        if category not in mapping:
            raise ValueError(f"Unknown measurement category: {category}")
        mapping[category](patch)

    # ===== final_report =====


    def update_final_report(self, patch: dict[str, Any]) -> None:
        self._deep_merge(self.data["final_report"], patch)