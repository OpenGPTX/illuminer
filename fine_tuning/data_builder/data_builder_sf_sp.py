from datasets import Dataset
from fine_tuning.dtos.config import DataConfig, PromptConfig, ModelConfig

from fine_tuning.data_builder.data_builder import DataBuilder

class DataBuilderSFSinglePrompt(DataBuilder):
    def __new__(cls, data_config: DataConfig, prompt_config: PromptConfig, model_config: ModelConfig, *args, **kwargs) -> Dataset:
        # create a new object
        obj = DataBuilder.__new__(cls, data_config=data_config, prompt_config=prompt_config, model_config=model_config)

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

            if obj.slot_dict:
                turn['slot_dict'] = "\n\t" + "\n\t".join([f"{k}: {v}," for k, v in candidate_slots.items()]) + "\n"
            turn['input'] = obj._fill_prompt(turn)
            turn['target'] = "{" + ", ".join([f"{slot}: {expected_slots[slot]}"
                                                if slot in expected_slots
                                                else f"{slot}: none"
                                                for slot in candidate_slots]) + "}"

        dataset = obj._preprocess(Dataset.from_list(obj.data))
        dataset = dataset.remove_columns(['utterance', 'domain', 'intent', 'slots', 'slot_dict'])

        return dataset