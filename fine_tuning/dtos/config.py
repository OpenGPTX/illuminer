"""
Schema definition and validation of the hierarchical config files.
"""
from hydra.core.config_store import ConfigStore
from pydantic import validator
from omegaconf import MISSING
from typing import List, Optional, Union

# we use pydantic as a replacement of default dataclasses for more validation features
from pydantic.dataclasses import dataclass

from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM

@dataclass
class MainConfig:
    model_output_dir: str

@dataclass
class TrainerConfig:
    output_dir: Optional[str] = "outputs/"
    learning_rate: Optional[float] = 1e-3
    lr_scheduler_type: Optional[str] = "linear"
    num_train_epochs: Optional[int] = 1
    max_steps: Optional[int] = -1
    warmup_steps: Optional[int] = 0
    logging_strategy: Optional[str] = "steps"
    logging_steps: Optional[int] = 10
    save_strategy: Optional[str] = "no"
    report_to: Optional[str] = "none"
    fp16: Optional[bool] = False
    auto_find_batch_size: Optional[bool] = False
    per_device_train_batch_size: Optional[int] = 8
    gradient_accumulation_steps: Optional[int] = 1
    remove_unused_columns: Optional[bool] = False
    run_name: Optional[str] = None  # updated later

@dataclass
class ModelConfig:
    model_name: str
    model_type: str
    device: Optional[str] = "auto"
    use_accelerate: Optional[bool] = True
    cache_dir: Optional[str] = None

    @validator('model_type')
    def check_model_type(cls, v: str) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
        if v == "AutoModelForCausalLM":
            return AutoModelForCausalLM
        elif v == "AutoModelForSeq2SeqLM":
            return AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Invalid model type \"{v}\"passed.")

@dataclass
class PeftConfig:
    peft_type: str
    task_type: Optional[str] = None                     # updated later
    freezing_original_weights: Optional[bool] = False
    r: Optional[int] = 16                               #LoRA
    lora_alpha: Optional[int] = 32                      #LoRA
    lora_dropout: Optional[float] = 0.05                #LoRA
    bias: Optional[str] = "none"                        #LoRA
    target_modules: Optional[List[str]] = None          # LoRA, IA3, updated later
    feedforward_modules: Optional[List[str]] = None     # IA3, updated later
    inference_mode: Optional[bool] = False              # PrefixTuning, PromptTuning
    num_virtual_tokens: Optional[int] = 20              # PrefixTuning, PromptTuning
    prompt_tuning_init: Optional[str] = None            # PromptTuning, updated later
    prompt_tuning_init_text : Optional[str] = None      # PromptTuning, updated later

@dataclass
class DataConfig:
    data_name: str
    data_path: str
    intent_desc_path: Optional[str] = None
    slot_desc_path: Optional[str] = None

@dataclass
class PromptConfig:
    data_builder: str
    prompt: str
    instruction: Optional[str] = None
    prompt_tuning_init_text: Optional[str] = None

@dataclass
class Config:
    main: MainConfig
    model: ModelConfig
    peft: PeftConfig
    trainer: TrainerConfig
    data: DataConfig = MISSING
    prompt: PromptConfig = MISSING

cs = ConfigStore.instance()
# name `base_config` is used for matching it with the main.yaml's default section
cs.store(name="base_config", node=Config)