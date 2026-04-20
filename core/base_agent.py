

from abc import abstractmethod


class BaseAgent:
    def __init__(self, role, presentation, llm_client, img_processing=False) -> None:
        self.role = role
        self.presentation = presentation   # 该角色知道的信息
        self.agent_hist = ""
        self.llm_client = llm_client  # Placeholder for the actual LLM client
        self.img_processing = img_processing  # to be set by child classes if needed
        self.patient_info_permission = role != "patient"    # TODO: 给 llm 的输入添加结构化的 patient 信息

        #==============


    def dialog_input(self, response: str) -> str:
        """Build the user message passed to the LLM."""
        return (
            f"Here is the history of your dialogue:\n"
            f"{self.agent_hist if self.agent_hist else '[No previous dialogue]'}\n\n"
            f"The user just said:\n{response}\n\n"
            f"Please continue the dialogue as {self.role}.\n"
        )
    
    @abstractmethod
    def system_prompt(self) -> str:
        raise NotImplementedError




    def inference(self, response: str) -> str:  # TODO: 如果开启了图像分析能力，response里可能包含一些特殊格式的图像信息（如INTAKE_IMAGE），需要在messages中添加图像信息 to be added
        messages = [
            {"role": "system", "content": self.system_prompt()},
            {"role": "user", "content": self.dialog_input(response)},
        ]
        return self.llm_client.think(messages, temperature=0)


    def add_hist(self, hist_str: str) -> None:
        self.agent_hist += hist_str + "\n\n"

