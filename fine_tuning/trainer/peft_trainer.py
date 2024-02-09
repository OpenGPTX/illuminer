import logging
import torch
import enum
from typing import Dict, List, Union
from hydra.utils import get_class
from torch import nn

from datasets import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
    AutoModelForCausalLM,
    DataCollatorForSeq2Seq,
    DataCollatorForLanguageModeling
)

from peft import get_peft_model, TaskType, LoraConfig, IA3Config, PrefixTuningConfig, PromptTuningConfig, \
    PromptTuningInit
from peft.utils.other import \
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING, \
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING, \
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

from fine_tuning.utils.util import load_tokenizer, load_model
from fine_tuning.dtos.config import ModelConfig, PeftConfig, TrainerConfig, PromptConfig

# A logger for this file
log = logging.getLogger(__name__)

class PeftTypes(str, enum.Enum):
    LORA = "LORA"
    IA3 = "IA3"
    PREFIX_TUNING = "PREFIX_TUNING"
    PROMPT_TUNING = "PROMPT_TUNING"

class CastOutputToFloat(nn.Sequential):
    def forward(self, x):
        return super().forward(x).to(torch.float32)

class PeftTrainer:
    def __init__(
        self,
        model_output_dir: str,
        model_config: ModelConfig,
        peft_config: PeftConfig,
        trainer_config: TrainerConfig,
        prompt_config: PromptConfig,
        dataset: Dataset
    ):
        self.dataset = dataset
        self.__model_config = model_config
        self.__peft_config = peft_config
        self.__trainer_config = trainer_config
        self.__prompt_config = prompt_config

        self.__model_type = get_class(f"transformers.{self.__model_config.model_type}")
        self.__model_name = self.__model_config.model_name.split("/")[-1]

        self.__model_output_dir = model_output_dir

    def _get_model_family(self):
        if "WizardLM" in self.__model_name or "vicuna" in self.__model_name:
            return "llama"
        else:
            return self.__model_name

    def _get_module_values(self,
            mapping_dict: Dict[str, List[str]]
    ) -> List[str]:
        model_family = self._get_model_family()
        for name in list(mapping_dict.keys()):
            if name in model_family:
                return mapping_dict[name]
        else:
            raise Exception(f"Can not find value for the model: {model_family}.")

    def _get_task_type(self):
        if not self.__peft_config.task_type:  # task_type is not in config
            task_type = TaskType.CAUSAL_LM if self.__model_type == AutoModelForCausalLM else TaskType.SEQ_2_SEQ_LM
        else:
            task_type = get_class(self.__peft_config.task_type)
        return task_type

    def _get_prompt_tuning_init(self):
        if not self.__prompt_config.prompt_tuning_init_text:
            return PromptTuningInit.TEXT
        else:
            return PromptTuningInit.RANDOM

    def get_peft_config(self) -> LoraConfig | IA3Config:
        if self.__peft_config.peft_type == PeftTypes.LORA:
            config = LoraConfig(
                task_type=self._get_task_type(),
                r=self.__peft_config.r,
                lora_alpha=self.__peft_config.lora_alpha,
                target_modules=self._get_module_values(
                    mapping_dict=TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING
                ),
                lora_dropout=self.__peft_config.lora_dropout,
                bias=self.__peft_config.bias,
            )

        elif self.__peft_config.peft_type == PeftTypes.IA3:
            config = IA3Config(
                task_type=self._get_task_type(),
                target_modules=self._get_module_values(
                    mapping_dict=TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING
                ),
                feedforward_modules=self._get_module_values(
                    mapping_dict=TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING
                )
            )

        elif self.__peft_config.peft_type == PeftTypes.PROMPT_TUNING:
            config = PromptTuningConfig(
                task_type=self._get_task_type(),
                num_virtual_tokens=self.__peft_config.num_virtual_tokens,
                inference_mode=self.__peft_config.inference_mode,
                prompt_tuning_init=self._get_prompt_tuning_init(),
                prompt_tuning_init_text=self.__prompt_config.prompt_tuning_init_text,
                tokenizer_name_or_path=self.__model_config.model_name
            )

        elif self.__peft_config.peft_type == PeftTypes.PREFIX_TUNING:
            config = PrefixTuningConfig(
                task_type=self._get_task_type(),
                num_virtual_tokens=self.__peft_config.num_virtual_tokens,
                inference_mode=self.__peft_config.inference_mode
            )

        else:
            raise Exception(f"Invalid PEFT type passed! Available types are: {PeftTypes._member_names_}.")

        log.info(f"PEFT config: {config}")
        return config

    def get_data_collator(self, model, tokenizer) -> Union[DataCollatorForLanguageModeling, DataCollatorForSeq2Seq]:
        def _get_seq2seq_lm_data_collator(model, tokenizer) -> DataCollatorForSeq2Seq:
            # we want to ignore tokenizer pad token in the loss
            label_pad_token_id = -100

            # Data collator
            return DataCollatorForSeq2Seq(
                model=model,
                tokenizer=tokenizer,
                label_pad_token_id=label_pad_token_id,
                pad_to_multiple_of=8
            )

        def _get_causal_lm_data_collator(tokenizer) -> DataCollatorForLanguageModeling:
            # Data collator
            return DataCollatorForLanguageModeling(
                tokenizer=tokenizer,
                mlm=False
            )

        if self.__model_type == AutoModelForCausalLM:
            return _get_causal_lm_data_collator(tokenizer=tokenizer)
        else:
            return _get_seq2seq_lm_data_collator(model=model, tokenizer=tokenizer)

    def train(self):
        model = load_model(self.__model_config)
        tokenizer = load_tokenizer(self.__model_config)

        if self.__peft_config.freezing_original_weights:
            # preparing for PEFT Training
            for param in model.parameters():
                # freeze the model - train adapters later
                param.requires_grad = False
                if param.ndim == 1:
                    # cast the small parameters (e.g. layer norm) to fp32 for
                    # stability
                    param.data = param.data.to(torch.float32)

            # reduce number of stored activations
            model.gradient_checkpointing_enable()
            model.enable_input_require_grads()

            model.lm_head = CastOutputToFloat(model.lm_head)

        model = get_peft_model(model=model, peft_config=self.get_peft_config())
        model.print_trainable_parameters()

        training_args = TrainingArguments(**self.__trainer_config)
        log.info(training_args)

        trainer = Trainer(
            model=model,
            train_dataset=self.dataset,
            args=training_args,
            data_collator=self.get_data_collator(model=model, tokenizer=tokenizer)
        )

        # silence the warnings, re-enable for inference!
        model.config.use_cache = False

        trainer.train()

        model.save_pretrained(self.__model_output_dir)
        tokenizer.save_pretrained(self.__model_output_dir)
        log.info(f"PEFT model is saved in \"{self.__model_output_dir}\".")

