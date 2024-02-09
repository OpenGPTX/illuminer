import os
import enum
from typing import Dict, List, Union
from pathlib import Path
from hydra.utils import get_class
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling

from peft import TaskType, LoraConfig, IA3Config
from peft.utils.other import \
    TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING, \
    TRANSFORMERS_MODELS_TO_IA3_TARGET_MODULES_MAPPING, \
    TRANSFORMERS_MODELS_TO_IA3_FEEDFORWARD_MODULES_MAPPING

from fine_tuning.dtos.config import ModelConfig, PeftConfig

class PeftTypes(str, enum.Enum):
    LORA = "LORA"
    IA3 = "IA3"

def get_run_name(config_choices: Dict[str, str]):
    return '_'.join([Path(config_choices['prompt']).stem,
                     Path(config_choices['data']).stem,
                     Path(config_choices['model']).stem,
                     Path(config_choices['peft']).stem]
                    )

def load_model(model_config: ModelConfig):
    auto_model = get_class(f"transformers.{model_config.model_type}")

    model_args = {
        "pretrained_model_name_or_path": model_config.model_name,
        # "torch_dtype": torch.float16,
        "load_in_8bit": False
    }

    # if "falcon" in model_config.model_name:
    #     model_args["trust_remote_code"] = True

    if model_config.use_accelerate:
        model_args["device_map"] = "auto"

    if model_config.cache_dir:
        model_args["cache_dir"] = model_config.cache_dir

    try:
        model = auto_model.from_pretrained(**model_args)
    except:
        raise ValueError(
            f"The passed model type: \"{model_config.model_type}\" "
            f"is not suitable for the model \"{model_config.model_name}\"."
            )

    if model_config.device:
        if model_config.use_accelerate:
            os.environ["CUDA_VISIBLE_DEVICES"] = model_config.device
            print("Using CUDA devices:", os.environ["CUDA_VISIBLE_DEVICES"])
        else:
            model = model.to(model_config.device)
            print("Using CUDA devices:", model_config.device)

    return model

def load_tokenizer(model_config: ModelConfig):
    tokenizer_args = {
        "pretrained_model_name_or_path": model_config.model_name,
        "use_fast": False,
        "legacy": False,
    }

    if model_config.cache_dir:
        tokenizer_args["cache_dir"] = model_config.cache_dir

    tokenizer = AutoTokenizer.from_pretrained(**tokenizer_args)

    if "falcon" in model_config.model_name:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer

