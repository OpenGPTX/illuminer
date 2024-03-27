import logging
import mlflow
import json
import re

from abc import abstractmethod
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

from evaluation.dtos.dto import EvalOutput, Result, EvalDataSF, SlotFillingInstance
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate import EvaluateLLM
from evaluation.utils import util

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


mlflow.set_tracking_uri("http://localhost:5006/")

class EvaluateSinglePromptSlotFilling(EvaluateLLM):

    def _get_slots_dict(self, slot_desc_path: str):
        slots_in = json.load(open(slot_desc_path, "r"))
        return slots_in

    def __init__(
        self,
        data_path: str,
        slot_desc_path: str,
        prompt: str,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=data_path, prompt=prompt, llm=llm, cfg=cfg)
        self.data = EvalDataSF.parse_file(data_path).data
        self.slot_desc_path = slot_desc_path
        self.slot_dict = self._get_slots_dict(slot_desc_path)

        self.__logger = logging.getLogger(self.__class__.__name__)
        experiment = mlflow.set_experiment(f"eval_{self.cfg.data.data_name}_slot_filling")
        run = mlflow.start_run(run_name=self.cfg.main.run_name)

        # Get experiment details
        self.__logger.info(f"Experiment_id: {experiment.experiment_id}")
        self.__logger.info(f"Experiment: {experiment.name}")
        self.__logger.info(f"Run: {run.info.run_name}")
        self.__logger.info(f"Artifact Location: {experiment.artifact_location}")

    def evaluate(self, data_start_index: int = 0, data_end_index: int = -1) -> EvalOutput:
        """
        
        """
        num_correct_slots = 0
        num_gold_slots = 0
        num_predicted_slots = 0
        num_hallucinated_slots = 0

        eval_out = []

        data = self.data[data_start_index:data_end_index]
        mlflow.log_param('num_data_points', len(data))
        mlflow.log_param('data_path', self.cfg.data.data_path)

        self.__logger.info("Starting evaluation")

        filled_prompts = []
        for turn in tqdm(data, desc="Preparing data"):

            # get possible slots to prompt given the intent in this turn
            candidate_slots = {}
            if turn.intent in self.slot_dict: candidate_slots.update(self.slot_dict[turn.intent])
            if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

            filled_prompt = self.fill_prompt(turn=turn, slots=candidate_slots)
            filled_prompts.append(filled_prompt)
        
        bs = 32
        prompts_batches = util.batch(data=filled_prompts, bs=bs)

        responses = []
        for prompts in tqdm(prompts_batches, total=len(filled_prompts)//bs, desc="Generating responses"):
            outputs = self.llm.run(prompts=prompts, max_new_tokens=150, split_lines=False)
            responses.extend(outputs)

        for turn, filled_prompt, response in zip(tqdm(data, desc="Evaluating responses"), filled_prompts, responses):
            # get possible slots to prompt given the intent in this turn
            candidate_slots = {}
            if turn.intent in self.slot_dict: candidate_slots.update(self.slot_dict[turn.intent])
            if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

            # get ground truth slots
            expected_slots = {}
            for s in turn.slots:
                expected_slots[s.slot_name] = s.slot_values
                num_gold_slots += 1

            if not response.endswith('}'): response += '}'
            response = response.lower().strip()

            predicted_slots = {}
            for slot_name, slot_desc in candidate_slots.items():    # find slot values in the response
                if slot_name in expected_slots:
                    gold_slot_values = expected_slots[slot_name]
                else:
                    gold_slot_values = []

                slot_pattern = re.compile(slot_name.lower() + r": (.+?)[,;\}]")
                slot_value = re.search(slot_pattern, response)
                if slot_value:
                    slot_value = slot_value.group(1)
                    if slot_value != "none":                              # the answer span is predicted
                        predicted_slots[slot_name] = slot_value
                        num_predicted_slots += 1

                        if self.check_response_span(
                                expected=gold_slot_values,
                                response=slot_value
                        ):                                                  # the answer span is correct
                            num_correct_slots += 1
                        else:
                            if slot_value.lower() not in turn.utterance:    # the answer span is NOT in the utterance --> hallucinate
                                num_hallucinated_slots += 1

            sorted_expected_slots = dict(sorted(expected_slots.items()))
            sorted_predicted_slots = dict(sorted(predicted_slots.items()))

            expected_slot_values = "; ".join([f"{k}: {v}" for k, v in sorted_expected_slots.items()])
            predicted_slot_values = "; ".join([f"{k}: {v}" for k, v in sorted_predicted_slots.items()])

            eval_out.append(
                Result(
                    utterance=turn.utterance,
                    filled_prompt=filled_prompt,
                    expected=expected_slot_values,
                    predicted=predicted_slot_values,
                    response=response
                )
            )

        file_path = f"evaluation/logs/eval_output_{self.cfg.main.run_name}.json"

        try:
            accuracy = num_correct_slots / num_gold_slots
            precision = num_correct_slots / num_predicted_slots
            recall = num_correct_slots / num_gold_slots
            f1 = (2 * precision * recall) / (precision + recall)
            hallucinate = num_hallucinated_slots / num_predicted_slots
        except ZeroDivisionError:
            accuracy = 0
            precision = 0
            recall = 0
            f1 = 0
            hallucinate = 0

        output = EvalOutput(
            model_name=self.llm.model_name,
            time=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
            prompt=self.prompt,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            result=eval_out
        )

        self.__logger.info("Finishing evaluation")
        self.__logger.info(f"Total slots: {num_gold_slots}")
        self.__logger.info(f"Correct slots: {num_correct_slots}")
        self.__logger.info(f"Accuracy: {accuracy}")
        self.__logger.info(f"Precision: {precision}")
        self.__logger.info(f"Recall: {recall}")
        self.__logger.info(f"F1: {f1}")
        self.__logger.info(f"Hallucinated: {hallucinate}")

        self.save(file_path=file_path, output=output.dict())
        self.__logger.info(f"Saved evaluation results at {file_path}")

        config_path = f"evaluation/logs/config_{self.cfg.main.run_name}.yaml"
        OmegaConf.save(self.cfg, config_path)
        mlflow.log_artifact(config_path)        

        mlflow.log_param('total_slots', num_gold_slots)
        mlflow.log_param('correct_slots', num_correct_slots)
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "hallucinated": hallucinate
            }
        )
        mlflow.log_artifact(file_path)
        mlflow.log_artifact(self.slot_desc_path)

        return output



    @abstractmethod
    def fill_prompt(self, turn: SlotFillingInstance, slots: dict) -> str:
        ...
