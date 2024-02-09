from typing import Optional, Union
from pydantic import BaseModel, validator
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


class OpenAILLMConfig(BaseModel):
    model: str


class HuggingFaceLLMConfig(BaseModel):
    model: str
    model_type: str
    tokenizer: str
    use_fast: Optional[bool]
    change_pad_token: Optional[bool]
    adapter: Optional[str]
    device: Optional[str]
    cache_dir: Optional[str]
    use_accelerate: Optional[bool]

    @validator('model_type')
    def replace_hyphen(cls, v):
        if v == "AutoModelForCausalLM":
            return AutoModelForCausalLM
        elif v == "AutoModelForSeq2SeqLM":
            return AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Invalid model type \"{v}\"passed.")


class NLULLMConfig(BaseModel):
    model: str
    model_type: str
    tokenizer: str
    use_fast: Optional[bool] = False
    change_pad_token: Optional[bool] = False
    domain_adapter: Optional[str]
    intent_adapter: str
    slot_adapter: str
    cache_dir: Optional[str] = ""
    device: Optional[str] = ""
    use_accelerate: Optional[bool] = True

    @validator('model_type')
    def replace_hyphen(cls, v):
        if v == "AutoModelForCausalLM":
            return AutoModelForCausalLM
        elif v == "AutoModelForSeq2SeqLM":
            return AutoModelForSeq2SeqLM
        else:
            raise ValueError(f"Invalid model type \"{v}\"passed.")