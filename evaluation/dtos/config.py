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
from llm.openai_llm import OpenAILLM

@dataclass
class MainConfig:
    run_name: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None

@dataclass
class DataConfig:
    data_name: str
    data_path: str
    start_index: Optional[int] = 0
    end_index: Optional[int] = -1
    intent_desc_path: Optional[str] = None
    slot_desc_path: Optional[str] = None
    intent_example_path: Optional[str] = None
    slot_example_path: Optional[str] = None
    domains: Optional[List[str]] = None

@dataclass
class PromptConfig:
    eval_mode: str
    prompt: Optional[str] = None
    prompt_with_answer: Optional[str] = None
    instruction: Optional[str] = None
    k_per_intent: Optional[int] = 1
    k_per_slot: Optional[int] = 1
    max_examples: Optional[int] = 10
    intent_prompt: Optional[str] = None                 # for DST pipeline
    intent_prompt_with_answer: Optional[str] = None     # for DST pipeline
    intent_instruction: Optional[str] = None            # for DST pipeline
    slot_prompt: Optional[str] = None                   # for DST pipeline
    slot_prompt_with_answer: Optional[str] = None       # for DST pipeline
    slot_instruction: Optional[str] = None              # for DST pipeline
    domain_prompt: Optional[str] = None                 # for DST pipeline
    domain_prompt_with_answer: Optional[str] = None     # for DST pipeline
    domain_instruction: Optional[str] = None            # for DST pipeline

@dataclass
class ModelConfig:
    model_name: str
    model_type: Optional[str] = None
    adapter: Optional[str] = None
    device: Optional[str] = "auto"
    use_accelerate: Optional[bool] = True
    use_fast: Optional[bool] = False
    change_pad_token: Optional[bool] = False
    cache_dir: Optional[str] = None
    intent_adapter: Optional[str] = None                # for DST pipeline
    slot_adapter: Optional[str] = None                  # for DST pipeline
    domain_adapter: Optional[str] = None                # for DST pipeline

    @validator('model_type')
    def check_model_type(cls, v: str) -> Union[AutoModelForCausalLM, AutoModelForSeq2SeqLM]:
        if v == "AutoModelForCausalLM":
            return AutoModelForCausalLM
        elif v == "AutoModelForSeq2SeqLM":
            return AutoModelForSeq2SeqLM
        elif v == "OpenAILLM":
            return OpenAILLM
        else:
            raise ValueError(f"Invalid model type \"{v}\" passed.")

@dataclass
class Config:
    main: MainConfig
    model: ModelConfig
    data: DataConfig = MISSING
    prompt: PromptConfig = MISSING

cs = ConfigStore.instance()
# name `base_config` is used for matching it with the main.yaml's default section
cs.store(name="base_config", node=Config)