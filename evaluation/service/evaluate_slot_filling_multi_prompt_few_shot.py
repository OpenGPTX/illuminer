import logging
import mlflow
import random

from evaluation.dtos.dto import SlotFillingInstance, EvalDataSF
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate_slot_filling_multi_prompt import EvaluateMultiPromptSlotFilling

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
    )


class EvaluateFewShotMultiPromptSlotFilling(EvaluateMultiPromptSlotFilling):
    def __init__(
        self,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(cfg.data.data_path, cfg.data.slot_desc_path, cfg.prompt.prompt, llm=llm, cfg=cfg)
        self.k_per_slot = cfg.prompt.k_per_slot
        self.max_examples = cfg.prompt.max_examples
        self.prompt_with_answer = cfg.prompt.prompt_with_answer
        self.example_path = cfg.data.slot_example_path

        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating Multi Prompt Few-shot Slot Filler")

        mlflow.log_param('prompt', self.prompt)
        mlflow.log_param('prompt_with_answer', self.prompt_with_answer)
        mlflow.log_param('k_per_slot', self.k_per_slot)
        mlflow.log_param('max_examples', self.max_examples)
        mlflow.log_param('model_name', self.llm.model_name)
        mlflow.log_param('mode', "multi-prompt-few-shot")

    def fill_prompt(self, turn: SlotFillingInstance, slot_desc: str):
        # get relevant slots given the intent in this turn
        candidate_slots = {}
        if turn.intent in self.slot_dict: candidate_slots.update(self.slot_dict[turn.intent])
        if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

        few_shot_data = self._generate_few_shot_data(self.example_path, intent=turn.intent)
        few_shot_instances = [self.prompt_with_answer.format(
            **fs.dict(),
            slot_desc=candidate_slots[fs.slots[0].slot_name],
            slot_value=fs.slots[0].slot_values[0]
        ) for fs in few_shot_data]

        return "\n".join(few_shot_instances) + '\n\n' + \
               self.prompt.format(**turn.dict(),
                                  slot_desc=slot_desc
                                  )

    def _generate_few_shot_data(self, example_path: str, intent: str):
        few_shot_data = []
        few_shot_examples = EvalDataSF.parse_file(example_path).data
        few_shot_examples = [turn for turn in few_shot_examples if turn.intent == intent]   # filter data only
        # with a certain intent

        samples_per_slot_name = {}
        for idx, sample in enumerate(few_shot_examples):
            for slot in sample.slots:
                if slot.slot_name not in samples_per_slot_name: samples_per_slot_name[slot.slot_name] = []
                samples_per_slot_name[slot.slot_name].append(idx)

        # get relevant slots given the intent in this turn
        candidate_slots = {}
        if intent in self.slot_dict: candidate_slots.update(self.slot_dict[intent])
        if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

        for slot_name in candidate_slots:
            if slot_name in samples_per_slot_name:
                for i in range(self.k_per_slot):
                    random_idx = random.choice(samples_per_slot_name[slot_name])
                    item = few_shot_examples[random_idx]
                    filled_slots = []
                    for slot in item.slots:
                        filled_slots.append(slot.slot_name)
                        if slot.slot_name == slot_name:
                            few_shot_data.append(                   # positive example
                                SlotFillingInstance(
                                    utterance=item.utterance,
                                    domain=item.domain,
                                    intent=item.intent,
                                    slots=[slot]
                                )
                            )
                    non_filled_slots = [n for n in candidate_slots if n not in filled_slots]
                    if non_filled_slots:
                        random_slot = random.choice(non_filled_slots)
                        few_shot_data.append(                           # negative example
                            SlotFillingInstance(
                                utterance=item.utterance,
                                domain=item.domain,
                                intent=item.intent,
                                slots=[{"slot_name": random_slot, "slot_values": ["none"]}]
                            )
                        )

        if len(few_shot_data) > self.max_examples:
            few_shot_data = random.sample(few_shot_data, k=self.max_examples)

        random.shuffle(few_shot_data)
        return few_shot_data
