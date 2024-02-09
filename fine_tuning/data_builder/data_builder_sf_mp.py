import random
from datasets import Dataset
from fine_tuning.dtos.config import DataConfig, PromptConfig, ModelConfig

from fine_tuning.data_builder.data_builder import DataBuilder

class DataBuilderSFMultiPrompt(DataBuilder):
    def __new__(cls, data_config: DataConfig, prompt_config: PromptConfig, model_config: ModelConfig, *args, **kwargs) -> Dataset:
        # create a new object
        obj = DataBuilder.__new__(cls, data_config=data_config, prompt_config=prompt_config, model_config=model_config)

        slots_data = []
        for turn in obj.data:
            # get possible slots to prompt given the intent in this turn
            candidate_slots = {}
            if obj.slot_dict:
                if turn['intent'] in obj.slot_dict: candidate_slots.update(obj.slot_dict[turn['intent']])
                if 'all' in obj.slot_dict: candidate_slots.update(obj.slot_dict['all'])

            # get ground truth slots
            expected_slots = {}
            for s in turn['slots']:
                expected_slots[s['slot_name']] = s['slot_values'][0]

            # negative examples, randomly sample the same amount as positive examples
            negative_slots = [slot_name for slot_name in candidate_slots if slot_name not in expected_slots]
            if len(negative_slots) > len(expected_slots):
                negative_slots = random.sample(negative_slots, len(expected_slots))

            for slot_name, slot_value in expected_slots.items():
                turn['slot_desc'] = candidate_slots[slot_name]
                slots_data.append({
                    'input': obj._fill_prompt(turn),
                    'target': slot_value
                })
            for slot_name in negative_slots:
                turn['slot_desc'] = candidate_slots[slot_name]
                slots_data.append({
                    'input': obj._fill_prompt(turn),
                    'target': 'none'
                })

        dataset = obj._preprocess(Dataset.from_list(slots_data))
        dataset = dataset.shuffle(seed=42)

        return dataset