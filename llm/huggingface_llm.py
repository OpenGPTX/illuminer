import logging
from typing import Union, Dict, List
from hydra.utils import get_class
from peft import PeftModel
from transformers import AutoTokenizer

from llm.llm import LLM
from evaluation.dtos.config import ModelConfig

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
    )

class HuggingFaceLLM(LLM):
    def __init__(
        self,
        cfg: ModelConfig
    ):
        self.__logger = logging.getLogger(self.__class__.__name__)
        self.__logger.info("Initiating HF Model Loading")

        self.__model_name = cfg.model_name
        self.__tokenizer_name = cfg.model_name

        self.__cache_dir = cfg.cache_dir
        self.__model_type = get_class(f"transformers.{cfg.model_type}")
        self.__device = cfg.device
        self.__use_accelerate = cfg.use_accelerate

        self.__use_fast = cfg.use_fast
        self.__change_pad_token = cfg.change_pad_token

        self.__adapter_name = cfg.adapter
        self.__intent_adapter = cfg.intent_adapter
        self.__slot_adapter = cfg.slot_adapter
        self.__domain_adapter = cfg.domain_adapter

        self.__model_args = self.__get_model_args()
        self.__tokenizer_args = self.__get_tokenizer_args()

        self.model = self.__load_model()
        self.tokenizer = self.__load_tokenizer()

        if self.__intent_adapter and self.__slot_adapter:   # for DST pipeline
            if self.__intent_adapter:  # Load the PEFT adapter for intent classification
                self.__logger.info("Load intent adapter...")
                self.__model_args.pop('pretrained_model_name_or_path', None)
                self.model = PeftModel.from_pretrained(model=self.model, model_id=self.__intent_adapter,
                                                       adapter_name='intent_classification', **self.__model_args)
                self.intent_model_name = self.__intent_adapter.split('/')[-1]

            if self.__domain_adapter:  # Load the PEFT adapter for domain classification
                self.__logger.info("Load domain adapter...")
                self.model.load_adapter(model_id=self.__domain_adapter, adapter_name='domain_classification')
                self.domain_model_name = self.__domain_adapter.split('/')[-1]

            if self.__slot_adapter:  # Load the PEFT adapter for slot filling
                self.__logger.info("Load slot adapter...")
                self.model.load_adapter(model_id=self.__slot_adapter, adapter_name='slot_filling')
                self.slot_model_name = self.__slot_adapter.split('/')[-1]
        else:
            self.intent_model_name = self.__model_name
            self.domain_model_name = self.__model_name
            self.slot_model_name = self.__model_name

            self.__logger.info("Adapters loaded.")

            self.model.eval()
            self.model.config.use_cache = True

    def __get_model_args(self) -> Dict[str, str]:
        model_args = {
            'pretrained_model_name_or_path': self.__model_name,
            # 'torch_dtype': torch.float16,
            'load_in_8bit': False
        }

        # if "falcon" in self.__model_name:
        #    model_args["trust_remote_code"] = True

        if self.__use_accelerate:
            model_args["device_map"] = "auto"

        if self.__cache_dir:
            model_args['cache_dir'] = self.__cache_dir

        return model_args

    def __get_tokenizer_args(self) -> Dict[str, str]:
        tokenizer_args = {
            'pretrained_model_name_or_path': self.__tokenizer_name,
            'use_fast': self.__use_fast,
            "legacy": False,
        }

        if "falcon" in self.__model_name or "vicuna" in self.__model_name or "polylm" in self.__model_name:
            tokenizer_args["padding_side"] = "left"

        if self.__cache_dir:
            tokenizer_args['cache_dir'] = self.__cache_dir

        return tokenizer_args

    def __load_model(self):
        try:
            model = self.__model_type.from_pretrained(**self.__model_args)
        except:
            raise ValueError(
                f"The passed model type: \"{self.__model_type.__name__}\" " \
                f"is not suitable for the model \"{self.__model_name}\"." \
            )

        if self.__adapter_name:
            model = PeftModel.from_pretrained(
                model=model,
                model_id=self.__adapter_name
            )

        if self.__device:
            if self.__use_accelerate:
                # os.environ["CUDA_VISIBLE_DEVICES"] = self.__device
                # print("Using CUDA devices:", os.environ["CUDA_VISIBLE_DEVICES"])
                pass
            else:
                print("loading model to", self.__device)
                model = model.to(self.__device)
                print("Using CUDA devices:", self.__device)

        return model

    def __load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(**self.__tokenizer_args)

        if tokenizer.pad_token_id is None or self.__adapter_name:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _postprocess_generated_ids(self, tokenized, generated_ids, split_lines: bool = True):
        new_ids = generated_ids
        if not self.model.config.is_encoder_decoder:
            # remove input ids from the genreated ids.
            # Note: This doesn't apply to AutoModelForSeq2SeqLM
            new_ids = generated_ids[:, tokenized.input_ids.shape[1]:]

        responses = self.tokenizer.batch_decode(new_ids, skip_special_tokens=True,  clean_up_tokenization_spaces=True)
        responses = [response.replace("<pad>", "") for response in responses]  # LoRA output somehow contains <pad>
        responses = [response.strip() for response in responses]

        if split_lines:
            responses_post = []
            for response in responses:
                if response:
                    responses_post.append(response.splitlines()[0])
                else:
                    responses_post.append(response)
            return responses_post

        return responses

    def run(self, prompts: Union[str, List[str]], max_new_tokens: int = 10, split_lines: bool = True) -> str:
        if self.__device and not self.__use_accelerate:
            tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.__device)
        else:
            tokenized = self.tokenizer(prompts, return_tensors="pt", padding=True, return_token_type_ids=False).to(0)

        generated_ids = self.model.generate(**tokenized, max_new_tokens=max_new_tokens)
        responses = self._postprocess_generated_ids(tokenized, generated_ids, split_lines)
        return responses

    def run_domain(self, prompts: Union[str, List[str]]) -> str:
        if self.domain_model_name != self.__model_name:
            self.model.set_adapter('domain_classification')
        responses = self.run(prompts, max_new_tokens=20, split_lines=True)
        return responses

    def run_intent(self, prompts: Union[str, List[str]]) -> str:
        if self.intent_model_name != self.__model_name:
            self.model.set_adapter('intent_classification')
        responses = self.run(prompts, max_new_tokens=20, split_lines=True)
        return responses

    def run_slot(self, prompts: Union[str, List[str]]) -> str:
        if self.slot_model_name != self.__model_name:
            self.model.set_adapter('slot_filling')
        responses = self.run(prompts, max_new_tokens=100, split_lines=False)
        return responses

