import os
import logging
import hydra
import mlflow
from omegaconf import OmegaConf
from hydra.core.hydra_config import HydraConfig
from hydra.utils import get_class

from evaluation.dtos.config import Config, ModelConfig
from evaluation.utils.util import get_run_name

# A logger for this file
log = logging.getLogger(__name__)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from evaluation.dtos.dto import \
    ZeroShotICEvalConfig, \
    FewShotICEvalConfig, \
    ZeroShotSFMPEvalConfig, \
    FewShotSFMPEvalConfig, \
    ZeroShotSFSPEvalConfig, \
    FewShotSFSPEvalConfig, \
    SingleTurnDSTEvalConfig
from llm.huggingface_llm import HuggingFaceLLM
from llm.openai_llm import OpenAILLM
from evaluation.service.evaluate_intent_classification_few_shot import \
    EvaluateFewShotIntentClassifier
from evaluation.service.evaluate_intent_classification_zero_shot import \
    EvaluateZeroShotIntentClassifier
from evaluation.service.evaluate_slot_filling_multi_prompt_few_shot import \
    EvaluateFewShotMultiPromptSlotFilling
from evaluation.service.evaluate_slot_filling_multi_prompt_zero_shot import \
    EvaluateZeroShotMultiPromptSlotFilling
from evaluation.service.evaluate_slot_filling_single_prompt_few_shot import \
    EvaluateFewShotSinglePromptSlotFilling
from evaluation.service.evaluate_slot_filling_single_prompt_zero_shot import \
    EvaluateZeroShotSinglePromptSlotFilling
from evaluation.service.evaluate_single_turn_dst import \
    EvaluateSingleTurnDST

def get_llm(model_config: ModelConfig):
    if model_config.model_type == "OpenAILLM":
        return OpenAILLM(model=model_config.model_name)
    else:
        return HuggingFaceLLM(model_config)


def get_eval_class(cfg: Config):
    try:
        eval_class = get_class(f"evaluation.dtos.dto.{cfg.prompt.eval_mode}")
        if eval_class == ZeroShotICEvalConfig:
            return EvaluateZeroShotIntentClassifier(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
        elif eval_class == FewShotICEvalConfig:
            return EvaluateFewShotIntentClassifier(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
        elif eval_class == ZeroShotSFSPEvalConfig:
            return EvaluateZeroShotSinglePromptSlotFilling(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
        elif eval_class == FewShotSFSPEvalConfig:
            return EvaluateFewShotSinglePromptSlotFilling(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
        elif eval_class == ZeroShotSFMPEvalConfig:
            return EvaluateZeroShotMultiPromptSlotFilling(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
        elif eval_class == FewShotSFMPEvalConfig:
            return EvaluateFewShotMultiPromptSlotFilling(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
        elif eval_class == SingleTurnDSTEvalConfig:
            return EvaluateSingleTurnDST(
                llm=get_llm(cfg.model),
                cfg=cfg
            )
    except:
        raise Exception(f"Invalid evaluation mode: {cfg.prompt.eval_mode}")

@hydra.main(config_path="configs", config_name="main", version_base="1.2")
def run(cfg: Config) -> None:
    # this line actually runs the checks of pydantic
    OmegaConf.to_object(cfg)

    # log config to console
    log.info(f"\n{OmegaConf.to_yaml(cfg)}")
    config_choices = HydraConfig.get().runtime.choices
    cfg.main.run_name = get_run_name(config_choices=config_choices)

    mlflow.set_tracking_uri(cfg.main.mlflow_tracking_uri)
    eval_class = get_eval_class(cfg)
    eval_class.evaluate(data_start_index=cfg.data.start_index, data_end_index=cfg.data.end_index)

if __name__ == '__main__':
    run()
