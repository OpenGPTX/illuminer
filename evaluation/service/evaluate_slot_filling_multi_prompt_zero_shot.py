import logging
import mlflow

from evaluation.dtos.dto import SlotFillingInstance
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate_slot_filling_multi_prompt import EvaluateMultiPromptSlotFilling

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
    )


class EvaluateZeroShotMultiPromptSlotFilling(EvaluateMultiPromptSlotFilling):
    def __init__(
        self,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=cfg.data.data_path, slot_desc_path=cfg.data.slot_desc_path, prompt=cfg.prompt.prompt,
                         llm=llm, cfg=cfg)
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating Multi Prompt Zero-shot Slot Filler")

        mlflow.log_param('prompt', self.prompt)
        mlflow.log_param('model_name', self.llm.model_name)
        mlflow.log_param('mode', "multi-prompt-zero-shot")

    def fill_prompt(self, turn: SlotFillingInstance, slot_desc: str):
        return self.prompt.format(
            utterance=turn.utterance,
            slot_desc=slot_desc
        )
