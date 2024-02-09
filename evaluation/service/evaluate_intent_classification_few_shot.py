import logging
import random
import mlflow

from evaluation.dtos.dto import IntentClassificationInstance, EvalDataIC
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate_intent_classification import EvaluateIntentClassifier

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


class EvaluateFewShotIntentClassifier(EvaluateIntentClassifier):
    def __init__(
        self,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=cfg.data.data_path, intent_desc_path=cfg.data.intent_desc_path,
                         prompt=cfg.prompt.prompt, llm=llm, cfg=cfg)
        self.k_per_intent = cfg.prompt.k_per_intent
        self.max_examples = cfg.prompt.max_examples
        self.prompt_with_answer = cfg.prompt.prompt_with_answer
        self.instruction = cfg.prompt.instruction
        self.example_path = cfg.data.intent_example_path

        self.examples_per_domain = {}
        self.examples_per_intent = {}
        if self.example_path:
            example_data = EvalDataIC.parse_file(self.example_path).data
            for item in example_data:
                if item.intent in self.intent_dict:
                    item.intent = self.intent_dict[item.intent]
                    if item.domain not in self.examples_per_domain:
                        self.examples_per_domain[item.domain] = []
                    if item.intent not in self.examples_per_intent:
                        self.examples_per_intent[item.intent] = []
                    self.examples_per_domain[item.domain].append(item)
                    self.examples_per_intent[item.intent].append(item)

        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating Few-shot Intent Classifer")

        if self.instruction:
            mlflow.log_param('instruction', self.instruction)
        mlflow.log_param('prompt', self.prompt)
        mlflow.log_param('prompt_with_answer', self.prompt_with_answer)
        mlflow.log_param('k_per_intent', self.k_per_intent)
        mlflow.log_param('max_examples', self.max_examples)
        mlflow.log_param('model_name', self.llm.model_name)
        mlflow.log_param('mode', "few-shot")

    def fill_prompt(self, turn: IntentClassificationInstance):
        if self.example_path:
            few_shot_data = self._generate_few_shot_data_from_training()
        else:
            few_shot_data = self._generate_few_shot_data()
        few_shot_instances = [self.prompt_with_answer.format(**fs.dict()) for fs in
                              few_shot_data]

        filled_prompt = ""
        if self.instruction:
            if self.intent_options:
                filled_prompt += self.instruction.format(
                    intent_options=" - " + " \n - ".join(self.intent_options)) + '\n'
            else:
                filled_prompt += self.instruction + '\n'
        filled_prompt += '\n'.join(few_shot_instances) + '\n'
        filled_prompt += self.prompt.format(**turn.dict())

        return filled_prompt

    def _generate_few_shot_data(self):
        few_shot_data = []

        for intent in self.intents:
            count = 0
            for item in self.data[-25:]:
                if item.intent == intent:
                    count += 1
                    if self.intent_dict:
                        item.intent = self.intent_dict[item.intent]
                    few_shot_data.append(item)
                if count == self.k_per_intent:
                    break
        return few_shot_data

    def _generate_few_shot_data_from_training(self):
        few_shot_data = []

        if self.cfg.data.domains:
            for key in self.examples_per_intent:
                for item in random.sample(self.examples_per_intent[key], k=self.k_per_intent):
                    few_shot_data.append(item)
        else: # no domain is specified, perform domain classification instead
            for key in self.examples_per_domain:
                for item in random.sample(self.examples_per_domain[key], k=self.k_per_intent):
                    few_shot_data.append(item)

        if len(few_shot_data) > self.max_examples:
            few_shot_data = random.sample(few_shot_data, k=self.max_examples)

        random.shuffle(few_shot_data)
        return few_shot_data
