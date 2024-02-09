import logging
import mlflow

from evaluation.dtos.dto import SlotFillingInstance
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate_slot_filling_single_prompt import EvaluateSinglePromptSlotFilling

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
    )


class EvaluateZeroShotSinglePromptSlotFilling(EvaluateSinglePromptSlotFilling):
    def __init__(
        self,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=cfg.data.data_path, slot_desc_path=cfg.data.slot_desc_path, prompt=cfg.prompt.prompt,
                         llm=llm, cfg=cfg)
        self.instruction = cfg.prompt.instruction
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating Single Prompt Zero-shot Slot Filler")

        if self.instruction:
            mlflow.log_param('instruction', self.instruction)
        mlflow.log_param('prompt', self.prompt)
        mlflow.log_param('model_name', self.llm.model_name)
        mlflow.log_param('mode', "single-prompt-zero-shot")

    def fill_prompt(self, turn: SlotFillingInstance, slots: dict):
        filled_prompt = ""
        if self.instruction:
            filled_prompt += self.instruction.format(
                slot_dict="\n\t" + "\n\t".join([f"{k}: {v}," for k, v in slots.items()]) + '\n')
            filled_prompt += self.prompt.format(**turn.dict())
        else:
            filled_prompt += self.prompt.format(
                slot_dict="\n\t" + "\n\t".join([f"{k}: {v}," for k, v in slots.items()]) + '\n',
                **turn.dict()
            )

        return filled_prompt
