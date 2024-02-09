import logging
import random
import mlflow

from evaluation.dtos.dto import SlotFillingInstance, EvalDataSF
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate_slot_filling_single_prompt import EvaluateSinglePromptSlotFilling

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
    )


class EvaluateFewShotSinglePromptSlotFilling(EvaluateSinglePromptSlotFilling):
    def __init__(
        self,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=cfg.data.data_path, slot_desc_path=cfg.data.slot_desc_path, prompt=cfg.prompt.prompt,
                         llm=llm, cfg=cfg)
        self.k_per_slot = cfg.prompt.k_per_slot
        self.max_examples = cfg.prompt.max_examples
        self.prompt_with_answer = cfg.prompt.prompt_with_answer
        self.instruction = cfg.prompt.instruction
        self.example_path = cfg.data.slot_example_path

        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating Single Prompt Few-shot Slot Filler")

        if self.instruction:
            mlflow.log_param('instruction', self.instruction)
        mlflow.log_param('prompt', self.prompt)
        mlflow.log_param('prompt_with_answer', self.prompt_with_answer)
        mlflow.log_param('k_per_slot', self.k_per_slot)
        mlflow.log_param('max_examples', self.max_examples)
        mlflow.log_param('model_name', self.llm.model_name)
        mlflow.log_param('mode', "single-prompt-few-shot")

    def fill_prompt(self, turn: SlotFillingInstance, slots: dict):
        few_shot_data = self._generate_few_shot_data(self.example_path, intent=turn.intent)
        few_shot_slots = [{f"{slot.slot_name}": f"{slot.slot_values[0]}" for slot in fs.slots} for fs in few_shot_data]

        few_shot_instances = [self.prompt_with_answer.format(
            utterance=fs.utterance,
            slots=", ".join([f"{slot}: {few_shot_slots[i][slot]}"
                             if slot in few_shot_slots[i]
                             else f"{slot}: none"
                             for slot in slots])
        ) for i, fs in enumerate(few_shot_data)]

        filled_prompt = ""
        if self.instruction:
            filled_prompt += self.instruction.format(
                slot_dict="\n\t" + "\n\t".join([f"{k}: {v}," for k, v in slots.items()]) + '\n')
            filled_prompt += '\n'.join(few_shot_instances) + '\n'
            filled_prompt += self.prompt.format(**turn.dict())
        else:
            filled_prompt += '\n'.join(few_shot_instances) + '\n'
            filled_prompt += self.prompt.format(
                slot_dict="\n\t" + "\n\t".join([f"{k}: {v}," for k, v in slots.items()]) + '\n',
                **turn.dict()
            )

        return filled_prompt

    def _generate_few_shot_data(self, example_path: str, intent: str):
        few_shot_data = []
        few_shot_examples = EvalDataSF.parse_file(example_path).data
        few_shot_examples = [turn for turn in few_shot_examples if turn.intent == intent]  # filter data only
        # with a certain intent

        samples_per_slot_name = {}
        for idx, sample in enumerate(few_shot_examples):
            for slot in sample.slots:
                if slot.slot_name not in samples_per_slot_name: samples_per_slot_name[slot.slot_name] = []
                samples_per_slot_name[slot.slot_name].append(idx)

        counts_per_slot_name = {}
        for slot_name in samples_per_slot_name: counts_per_slot_name[slot_name] = self.k_per_slot

        for slot_name in samples_per_slot_name:
            for count in range(counts_per_slot_name[slot_name]):
                random_idx = random.choice(samples_per_slot_name[slot_name])
                random_sample = few_shot_examples[random_idx]
                few_shot_data.append(random_sample)

                for slot in random_sample.slots:
                    counts_per_slot_name[slot.slot_name] -= 1

        if len(few_shot_data) > self.max_examples:
            few_shot_data = random.sample(few_shot_data, k=self.max_examples)

        random.shuffle(few_shot_data)
        return few_shot_data
