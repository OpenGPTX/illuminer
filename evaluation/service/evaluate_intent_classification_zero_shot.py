import logging
import mlflow

from evaluation.dtos.dto import IntentClassificationInstance
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate_intent_classification import EvaluateIntentClassifier

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


class EvaluateZeroShotIntentClassifier(EvaluateIntentClassifier):
    def __init__(
        self,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=cfg.data.data_path, intent_desc_path=cfg.data.intent_desc_path,
                         prompt=cfg.prompt.prompt, llm=llm, cfg=cfg)
        self.instruction = cfg.prompt.instruction
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating Zero-shot Intent Classifer")

        if self.instruction:
            mlflow.log_param('instruction', self.instruction)
        mlflow.log_param('prompt', self.prompt)
        mlflow.log_param('model_name', self.llm.model_name)
        mlflow.log_param('mode', "zero-shot")

    def fill_prompt(self, turn: IntentClassificationInstance):
        filled_prompt = ""
        if self.instruction:
            if self.intent_options:
                filled_prompt += self.instruction.format(
                    intent_options=" - " + " \n - ".join(self.intent_options)) + '\n'
            else:
                filled_prompt += self.instruction + '\n'
        filled_prompt += self.prompt.format(**turn.dict())

        return filled_prompt

