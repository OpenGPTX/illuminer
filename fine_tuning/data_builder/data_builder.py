from typing import Dict
from datasets import Dataset
from fine_tuning.dtos.config import DataConfig, PromptConfig, ModelConfig
from fine_tuning.utils.util import load_tokenizer
import json
from transformers import AutoModelForSeq2SeqLM
from hydra.utils import get_class

class DataBuilder:
    def __new__(cls, data_config: DataConfig, prompt_config: PromptConfig, model_config: ModelConfig):
        # create a new object
        obj = super().__new__(cls)

        obj.__data_config = data_config
        obj.__prompt_config = prompt_config
        obj.__model_config = model_config

        obj.data = json.load(open(obj.__data_config.data_path))['data']

        if obj.__data_config.intent_desc_path:
            obj.intent_dict = json.load(open(obj.__data_config.intent_desc_path))
            obj.intent_labels = {k: v for d in obj.intent_dict for k, v in obj.intent_dict[d].items()}
            obj.intent_options = [v for d in obj.intent_dict for k, v in obj.intent_dict[d].items()]
            obj.domain_options = [d for d in obj.intent_dict]

        if obj.__data_config.slot_desc_path:
            obj.slot_dict = json.load(open(obj.__data_config.slot_desc_path))

        return obj

    def _fill_prompt(self, turn: Dict) -> str:
        filled_prompt = ""
        if self.__prompt_config.instruction:
            filled_prompt += self.__prompt_config.instruction.format(**turn)
        filled_prompt += self.__prompt_config.prompt.format(**turn)

        return filled_prompt

    def _preprocess(self, dataset: Dataset) -> Dataset:
        tokenizer = load_tokenizer(self.__model_config)

        def __preprocess_seq2seq(examples):
            inputs = examples["input"]
            targets = examples["target"]

            # Compute max_length for inputs and targets
            max_length = max([len(tokenizer(input)["input_ids"]) for input in inputs])
            target_max_length = max([len(tokenizer(target)["input_ids"]) for target in targets])

            model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True,
                                     return_tensors="pt")
            labels = tokenizer(targets, max_length=target_max_length, padding="max_length", truncation=True,
                               return_tensors="pt")

            # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
            # padding in the loss.
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
            model_inputs["labels"] = labels["input_ids"]

            return model_inputs

        def __preprocess_input(examples):
            # input is basically the concatenation of input and target
            inputs = [f"{input_} {target_}" for input_, target_ in zip(examples["input"], examples["target"])]
            model_inputs = tokenizer(inputs)

            return model_inputs

        if get_class(f"transformers.{self.__model_config.model_type}") == AutoModelForSeq2SeqLM:
            tokenized_datasets = dataset.map(
                __preprocess_seq2seq,
                batched=True,
                remove_columns=['input', 'target'],
            )
        else:
            tokenized_datasets = dataset.map(
                __preprocess_input,
                batched=True,
                remove_columns=['input', 'target'],
            )

        return tokenized_datasets



