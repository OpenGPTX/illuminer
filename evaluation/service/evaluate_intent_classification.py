import logging
import mlflow
import json
import re

from abc import abstractmethod
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

from evaluation.dtos.dto import EvalOutput, Result, EvalDataIC, IntentClassificationInstance
from evaluation.dtos.config import Config
from llm.llm import LLM
from evaluation.service.evaluate import EvaluateLLM
from evaluation.utils import util

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


class EvaluateIntentClassifier(EvaluateLLM):

    def _get_intents_dict(self, intent_desc_path: str):
        intent_dict = json.load(open(intent_desc_path, "r"))
        return intent_dict

    def __init__(
        self,
        data_path: str,
        intent_desc_path: str,
        prompt: str,
        llm: LLM,
        cfg: Config
    ):
        super().__init__(data_path=data_path, prompt=prompt, llm=llm, cfg=cfg)
        self.data = EvalDataIC.parse_file(data_path).data
        self.intent_desc_path = intent_desc_path

        # Filter data on specific domain
        if cfg.data.domains:
            self.data = [d for d in self.data if d.domain in cfg.data.domains]

        # Get intent description
        if cfg.data.domains:
            self.intent_dict = {}
            for domain, intent_map in self._get_intents_dict(intent_desc_path).items():
                if domain in cfg.data.domains:
                    for k, v in intent_map.items():
                        self.intent_dict[k] = v

        else:   # no domain is specified, perform domain classification instead
            self.intent_dict = {}
            for domain in self._get_intents_dict(intent_desc_path):
                self.intent_dict[domain] = domain

        self.intent_dict_inv = {v.lower(): k for k, v in self.intent_dict.items()}
        self.intents = [k for k, v in self.intent_dict.items()]
        self.intent_options = [v for k, v in self.intent_dict.items()]

        self.__logger = logging.getLogger(self.__class__.__name__)
        experiment = mlflow.set_experiment(f"eval_{self.cfg.data.data_name}_intent_classifier")
        run = mlflow.start_run(run_name=self.cfg.main.run_name)

        # Get experiment details
        self.__logger.info(f"Experiment_id: {experiment.experiment_id}")
        self.__logger.info(f"Experiment: {experiment.name}")
        self.__logger.info(f"Run: {run.info.run_name}")
        self.__logger.info(f"Artifact Location: {experiment.artifact_location}")

    def map_response_to_labels(self, response):
        if response.endswith("."): response = response[:-1]
        response = response.lower().strip()
        response = re.sub(r'[^-\w\s\(\)]', '', response)
        if response in self.intent_dict_inv:
            return self.intent_dict_inv[response]
        else:
            for intent_opt in self.intent_dict_inv:
                if intent_opt in response:
                    return self.intent_dict_inv[intent_opt]
        return 'none'

    def evaluate(self, data_start_index: int = 0, data_end_index: int = -1) -> EvalOutput:
        """
        
        """
        count = 0
        eval_out = []

        data = self.data[data_start_index:data_end_index]
        mlflow.log_param('num_data_points', len(data))
        mlflow.log_param('data_path', self.cfg.data.data_path)

        if self.cfg.data.domains:
            mlflow.log_param('task', 'intent')
        else:  # no domain is specified, perform domain classification instead
            mlflow.log_param('task', 'domain')

        y_true = []
        y_pred = []


        self.__logger.info("Starting evaluation")

        filled_prompts = []
        for turn in tqdm(data, desc="Preparing data"):
            filled_prompt = self.fill_prompt(turn=turn)
            filled_prompts.append(filled_prompt)

        bs = 32
        prompts_batches = util.batch(data=filled_prompts, bs=bs)

        responses = []
        for prompts in tqdm(prompts_batches, total=len(filled_prompts)//bs, desc="Generating responses"):
            outputs = self.llm.run(prompts=prompts, max_new_tokens=20)
            responses.extend(outputs)
                

        for turn, filled_prompt, response in zip(tqdm(data, desc="Evaluating responses"), filled_prompts, responses):
            if "," in response: response = response.split(",")[0]
            if "." in response: response = response.split(".")[0]
            original_response = response

            # map response to intent labels if necessary
            if self.intent_dict and self.intent_dict_inv:
                response = self.map_response_to_labels(response)

            if self.cfg.data.domains:
                expected = turn.intent
            else:  # no domain is specified, perform domain classification instead
                expected = turn.domain

            if self.check_response(
                expected=expected,
                expected_options=self.intents,
                response=response
            ):
                count += 1

            y_true.append(expected)
            y_pred.append(response)

            eval_out.append(
                Result(
                    utterance=turn.utterance,
                    filled_prompt=filled_prompt,
                    expected=f"{self.intent_dict[expected]} [{expected}]",
                    predicted=f"{response}",
                    response=f"{original_response}",
                )
            )

        file_path = f"evaluation/logs/eval_output_{self.cfg.main.run_name}.json"

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro', labels=list(set(y_true)))
        recall = recall_score(y_true, y_pred, average='macro', labels=list(set(y_true)))
        f1 = f1_score(y_true, y_pred, average='macro', labels=list(set(y_true)))

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
        self.__logger.info(f"Total count: {len(data)}")
        self.__logger.info(f"Correct count: {count}")
        self.__logger.info(f"Accuracy: {accuracy}")
        self.__logger.info(f"Precision: {precision}")
        self.__logger.info(f"Recall: {recall}")
        self.__logger.info(f"F1: {f1}")

        self.save(file_path=file_path, output=output.dict())
        self.__logger.info(f"Saved evaluation results at {file_path}")

        mlflow.log_param('correct_count', count)
        mlflow.log_metrics(
            {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1
            }
        )
        mlflow.log_artifact(file_path)
        mlflow.log_artifact(self.intent_desc_path)

        config_path = f"evaluation/logs/config_{self.cfg.main.run_name}.yaml"
        OmegaConf.save(self.cfg, config_path)
        mlflow.log_artifact(config_path)

        # Save the confusion matrix
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred,
                                                       labels=list(set(y_true)),
                                                       cmap=plt.cm.Blues,
                                                       xticks_rotation='vertical',
                                                       normalize="true")

        png_path = f"evaluation/logs/confusion_{self.cfg.main.run_name}.png"
        fig = disp.figure_
        if len(list(set(y_true))) > 50:
            fig.set_figwidth(25)
            fig.set_figheight(25)
        elif len(list(set(y_true))) > 30:
            fig.set_figwidth(20)
            fig.set_figheight(20)
        elif len(list(set(y_true))) > 10:
            fig.set_figwidth(15)
            fig.set_figheight(15)
        else:
            fig.set_figwidth(10)
            fig.set_figheight(10)

        fig.savefig(png_path)
        mlflow.log_artifact(png_path)

        return output

    @abstractmethod
    def fill_prompt(self, turn: IntentClassificationInstance) -> str:
        ...
