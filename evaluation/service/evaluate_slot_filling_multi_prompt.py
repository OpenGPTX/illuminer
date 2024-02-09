import logging
import mlflow
import json

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
    level=logging.INFO,
    )


class EvaluateMultiPromptSlotFilling(EvaluateLLM):

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

        utterances = []
        filled_prompts = []
        gold_slot_values = []
        for turn in tqdm(data, desc="Preparing data"):

            # get possible slots to prompt given the intent in this turn
            candidate_slots = {}
            if turn.intent in self.slot_dict: candidate_slots.update(self.slot_dict[turn.intent])
            if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

            # get ground truth slots
            expected_slots = {}
            for s in turn.slots:
                expected_slots[s.slot_name] = s.slot_values
                num_gold_slots += 1

                if s.slot_name not in candidate_slots:  # ground truth slot is not even considered as
                    # one of the possible slots given the intent in this turn
                    eval_out.append(
                        Result(
                            utterance=turn.utterance,
                            filled_prompt="",
                            expected=f"{s.slot_name}: {','.join(s.slot_values)}; ",
                            predicted="",
                            response=""
                        )
                    )

            for slot_name, slot_desc in candidate_slots.items():    # prompting based on candidate slots
                # given the intent
                utterances.append(turn.utterance)
                filled_prompt = self.fill_prompt(turn=turn, slot_desc=slot_desc)
                filled_prompts.append(filled_prompt)

                if slot_name in expected_slots:
                    gold_slot_values.append((slot_name, expected_slots[slot_name]))
                else:
                    gold_slot_values.append((slot_name, []))

        bs = 64
        prompts_batches = util.batch(data=filled_prompts, bs=bs)

        responses = []
        for prompts in tqdm(prompts_batches, total=len(filled_prompts) // bs, desc="Generating responses"):
            outputs = self.llm.run(prompts=prompts, max_new_tokens=100)
            responses.extend(outputs)

        for utterance, (slot_name, slot_value), filled_prompt, response in zip(tqdm(
                utterances, desc="Evaluating responses"), gold_slot_values, filled_prompts, responses):

            original_response = response
            if "," in response: response = response.split(",")[0]
            if "." in response: response = response.split(".")[0]

            correct = False

            if not response.startswith("none"):     # the answer span is predicted
                num_predicted_slots += 1

                if self.check_response_span(
                        expected=slot_value,
                        response=response
                ):                                  # the answer span is correct
                    num_correct_slots += 1
                    correct = True
                else:
                    if not self.check_response_span(
                        expected=slot_value,
                        response=response
                    ):                              # the answer span is NOT in the utterance --> hallucinate
                        num_hallucinated_slots += 1


                eval_out.append(
                    Result(
                        utterance=utterance,
                        filled_prompt=filled_prompt,
                        expected=f"{slot_name}: [{','.join(slot_value)}]" if slot_value else f"{slot_name}: none",
                        predicted=f"{slot_name}: {response} [{correct}]",
                        response=original_response
                    )
                )
            else:
                if slot_value:
                    eval_out.append(
                        Result(
                            utterance=utterance,
                            filled_prompt=filled_prompt,
                            expected=f"{slot_name}: [{','.join(slot_value)}]" if slot_value else f"{slot_name}: none",
                            predicted=f"{slot_name}: none",
                            response=original_response
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
    def fill_prompt(self, turn: SlotFillingInstance, slot_desc: str) -> str:
        ...
