# TODO: 增加多agent的config信息 (not urgent)


# intake 对话轮数


class SysConfig:
    def __init__(self, max_infs: int = 20, img_processing: bool = False):
        self.max_infs = max_infs
        self.img_processing = img_processing
        