import os
import logging
import hydra
from omegaconf import OmegaConf
from pathlib import Path
from typing import List, Dict
from hydra.core.hydra_config import HydraConfig

from datasets import Dataset

from fine_tuning.dtos.config import Config, DataConfig, PromptConfig, ModelConfig
from fine_tuning.data_builder.data_builder_ic import DataBuilderIC
from fine_tuning.data_builder.data_builder_dc import DataBuilderDC
from fine_tuning.data_builder.data_builder_sf_sp import DataBuilderSFSinglePrompt
from fine_tuning.data_builder.data_builder_sf_mp import DataBuilderSFMultiPrompt
from fine_tuning.trainer.peft_trainer import PeftTrainer
from fine_tuning.utils.util import get_run_name

# A logger for this file
log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def run_trainer(dataset: Dataset, config: Config, config_choices: Dict[str, str]):
    config.trainer.run_name = get_run_name(config_choices)
    config.main.model_output_dir = str(Path(config.main.model_output_dir).joinpath(
                                        Path(config_choices['prompt'],
                                        config.trainer.run_name)
                                        ))
    trainer = PeftTrainer(
        model_output_dir=config.main.model_output_dir,
        model_config=config.model,
        peft_config=config.peft,
        trainer_config=config.trainer,
        prompt_config=config.prompt,
        dataset=dataset
    )
    trainer.train()

def build_dataset(data_cfg: DataConfig, prompt_cfg: PromptConfig, model_cfg: ModelConfig):
    if prompt_cfg.data_builder == "DataBuilderIC":
        return DataBuilderIC(data_cfg, prompt_cfg, model_cfg)

    elif prompt_cfg.data_builder == "DataBuilderDC":
        return DataBuilderDC(data_cfg, prompt_cfg, model_cfg)

    elif prompt_cfg.data_builder == "DataBuilderSFSinglePrompt":
        return DataBuilderSFSinglePrompt(data_cfg, prompt_cfg, model_cfg)

    elif prompt_cfg.data_builder == "DataBuilderSFMultiPrompt":
        return DataBuilderSFMultiPrompt(data_cfg, prompt_cfg, model_cfg)

    else:
        raise ValueError(f"Invalid data builder \"{prompt_cfg.data_builder}\" passed.")

@hydra.main(config_path="configs", config_name="main", version_base="1.2")
def run(cfg: Config) -> None:
    # this line actually runs the checks of pydantic
    OmegaConf.to_object(cfg)

    # log config to console
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    config_choices = HydraConfig.get().runtime.choices

    log.info(f"Build dataset for fine-tuning...")
    dataset = build_dataset(cfg.data, cfg.prompt, cfg.model)
    log.info(dataset)

    log.info(f"Run PEFT trainer...")
    run_trainer(dataset=dataset, config=cfg, config_choices=config_choices)


if __name__ == '__main__':
    run()
