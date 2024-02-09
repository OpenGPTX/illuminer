from datasets import Dataset
from fine_tuning.dtos.config import DataConfig, PromptConfig, ModelConfig

from fine_tuning.data_builder.data_builder import DataBuilder

class DataBuilderDC(DataBuilder):
    def __new__(cls, data_config: DataConfig, prompt_config: PromptConfig, model_config: ModelConfig, *args, **kwargs) -> Dataset:
        # create a new object
        obj = DataBuilder.__new__(cls, data_config=data_config, prompt_config=prompt_config, model_config=model_config)

        for turn in obj.data:
            if obj.domain_options:
                turn['domain_options'] = " - " + " \n - ".join(obj.domain_options) + "\n"
            turn['input'] = obj._fill_prompt(turn)
            turn['target'] = turn['domain']

        dataset = obj._preprocess(Dataset.from_list(obj.data))
        dataset = dataset.remove_columns(['utterance', 'domain', 'intent', 'domain_options'])

        return dataset