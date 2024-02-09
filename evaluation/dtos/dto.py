from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field

from llm.dtos import HuggingFaceLLMConfig, OpenAILLMConfig, NLULLMConfig


class DatasetConfig(BaseModel):
    name: str
    path: str
    start_index: Optional[int]
    end_index: Optional[int]


class ZeroShotICEvalConfig(BaseModel):
    mode: Literal["ZeroShotICEvalConfig"]
    prompt: str
    intent_desc_path: str
    instruction: Optional[str]
    domains: Optional[List[str]]


class ZeroShotSFMPEvalConfig(BaseModel):
    mode: Literal["ZeroShotSFMPEvalConfig"]
    prompt: str
    slot_desc_path: str
    slot_name: Optional[str]
    slot_desc: Optional[str]


class ZeroShotSFSPEvalConfig(BaseModel):
    mode: Literal["ZeroShotSFSPEvalConfig"]
    instruction: Optional[str]
    prompt: str
    slot_desc_path: str
    slot_name: Optional[str]
    slot_desc: Optional[str]


class ZeroShotACEvalConfig(BaseModel):
    mode: Literal["ZeroShotACEvalConfig"]
    instruction: str
    prompt: str
    act_desc_path: str


class FewShotICEvalConfig(BaseModel):
    mode: Literal["FewShotICEvalConfig"]
    prompt: str
    prompt_with_answer: str
    k_per_intent: int
    intent_desc_path: str
    example_path: Optional[str]
    instruction: Optional[str]
    domains: Optional[List[str]]


class FewShotSFMPEvalConfig(BaseModel):
    mode: Literal["FewShotSFMPEvalConfig"]
    prompt: str
    prompt_with_answer: str
    k_per_slot: int
    slot_desc_path: str
    example_path: Optional[str]
    slot_name: Optional[str]
    slot_desc: Optional[str]


class FewShotSFSPEvalConfig(BaseModel):
    mode: Literal["FewShotSFSPEvalConfig"]
    instruction: str
    prompt: str
    prompt_with_answer: str
    k_per_slot: int
    slot_desc_path: str
    example_path: Optional[str]
    slot_name: Optional[str]
    slot_desc: Optional[str]


class FewShotACEvalConfig(BaseModel):
    mode: Literal["FewShotACEvalConfig"]
    instruction: str
    prompt: str
    prompt_with_answer: str
    k_per_act: int
    act_desc_path: str
    example_path: Optional[str]


class SingleTurnDSTEvalConfig(BaseModel):
    mode: Literal["SingleTurnDSTEvalConfig"]
    domain_instruction: Optional[str]
    intent_instruction: str
    slot_instruction: str
    domain_prompt: Optional[str]
    domain_prompt_with_answer: Optional[str]
    intent_prompt: str
    intent_prompt_with_answer: Optional[str]
    slot_prompt: str
    slot_prompt_with_answer: Optional[str]
    intent_desc_path: str
    slot_desc_path: str
    intent_example_path: Optional[str]
    slot_example_path: Optional[str]
    k_per_intent: Optional[int]
    k_per_slot: Optional[int]


class ServiceConfig(BaseModel):
    dataset: DatasetConfig
    eval: Union[SingleTurnDSTEvalConfig,
                FewShotACEvalConfig, ZeroShotACEvalConfig,
                FewShotSFSPEvalConfig, ZeroShotSFSPEvalConfig,
                FewShotSFMPEvalConfig, FewShotICEvalConfig,
                ZeroShotSFMPEvalConfig, ZeroShotICEvalConfig] = Field(..., discriminator="mode")
    llm: Union[NLULLMConfig, HuggingFaceLLMConfig, OpenAILLMConfig]
    cfg_name: str


class IntentClassificationInstance(BaseModel):
    utterance: str
    domain: str
    intent: str


class EvalDataIC(BaseModel):
    data: List[IntentClassificationInstance]
    domains: Optional[List[str]]
    intents: Optional[List[str]]


class SlotInstance(BaseModel):
    slot_name: str
    slot_values: List[str]


class SlotFillingInstance(BaseModel):
    utterance: str
    domain: str
    intent: str
    slots: List[SlotInstance]


class EvalDataSF(BaseModel):
    data: List[SlotFillingInstance]


class ActClassificationInstance(BaseModel):
    utterance: str
    acts: List[str]


class EvalDataAC(BaseModel):
    data: List[ActClassificationInstance]
    dialog_acts: Optional[List[str]]


class DSTInstance(BaseModel):
    previous: str
    utterance: str
    acts: List[str]
    domain: str
    intent: str
    inform_slots: List[SlotInstance]
    request_slots: List[SlotInstance]
    state: List[SlotInstance]


class EvalDataDST(BaseModel):
    data: List[DSTInstance]


class Result(BaseModel):
    utterance: str
    filled_prompt: str
    expected: str
    predicted: str
    response: str


class EvalOutput(BaseModel):
    model_name: str
    time: str
    prompt: str
    accuracy: float
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    result: List[Result]


class EvalDSTOutput(BaseModel):
    model_name: str
    time: str
    domain_instruction: Optional[str]
    domain_prompt: Optional[str]
    intent_instruction: str
    intent_prompt: str
    slot_instruction: str
    slot_prompt: str
    domain_accuracy: float
    domain_precision: Optional[float]
    domain_recall: Optional[float]
    domain_f1: Optional[float]
    intent_accuracy: float
    intent_precision: Optional[float]
    intent_recall: Optional[float]
    intent_f1: Optional[float]
    avg_slot_accuracy: float
    slot_precision: Optional[float]
    slot_recall: Optional[float]
    slot_f1: Optional[float]
    slot_hallucinate: Optional[float]
    result: List[Result]
