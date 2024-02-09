import logging
import mlflow
import json
import re
import numpy as np
import random

from abc import abstractmethod
from datetime import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

from evaluation.dtos.dto import EvalDSTOutput, Result, EvalDataIC, EvalDataSF, SlotFillingInstance
from evaluation.dtos.config import Config
from llm.huggingface_llm import HuggingFaceLLM
from evaluation.service.evaluate import EvaluateLLM
from evaluation.utils import util

logging.basicConfig(
    format='[%(levelname)s] %(asctime)s - %(name)s - %(message)s',
    level=logging.INFO
)


class EvaluateSingleTurnDST(EvaluateLLM):

    def _get_intents_dict(self, intent_desc_path: str):
        intent_dict = json.load(open(intent_desc_path, "r"))
        return intent_dict

    def _get_slots_dict(self, slot_desc_path: str):
        slots_in = json.load(open(slot_desc_path, "r"))
        return slots_in

    def _get_intent_options(self, domain: str = ""):
        if domain in self.intent_dict:
            return [v for k, v in self.intent_dict[domain].items()]
        else:
            intent_options = []
            for d in self.intent_dict:
                intent_options += [v for k, v in self.intent_dict[d].items()]
            return intent_options

    def _get_intent_dict_inv(self):
        intent_dict_inv = {}
        for d in self.intent_dict:
            for k, v in self.intent_dict[d].items():
                intent_dict_inv[v] = k
        return intent_dict_inv

    def _get_domain_dict_inv(self):
        domain_dict_inv = {}
        for d in self.intent_dict:
            for k in self.intent_dict[d]:
                domain_dict_inv[k] = d
        return domain_dict_inv

    def __init__(
        self,
        llm: HuggingFaceLLM,
        cfg: Config
    ):
        super().__init__(data_path=cfg.data.data_path, prompt=cfg.prompt.prompt, llm=llm, cfg=cfg)

        self.llm = llm
        self.cfg = cfg

        self.data = EvalDataSF.parse_file(cfg.data.data_path).data

        self.intent_desc_path = cfg.data.intent_desc_path
        self.intent_dict = self._get_intents_dict(cfg.data.intent_desc_path)
        self.intent_dict_inv = self._get_intent_dict_inv()

        self.slot_desc_path = cfg.data.slot_desc_path
        self.slot_dict = self._get_slots_dict(cfg.data.slot_desc_path)

        self.intent_example_path = cfg.data.intent_example_path
        self.slot_example_path = cfg.data.slot_example_path

        self.examples_per_domain = {}
        self.examples_per_intent = {}
        if self.intent_example_path:
            example_data = EvalDataIC.parse_file(self.intent_example_path).data
            for item in example_data:
                if item.intent in self.intent_dict:
                    item.intent = self.intent_dict[item.intent]
                    if item.domain not in self.examples_per_domain:
                        self.examples_per_domain[item.domain] = []
                    if item.intent not in self.examples_per_intent:
                        self.examples_per_intent[item.intent] = []
                    self.examples_per_domain[item.domain].append(item)
                    self.examples_per_intent[item.intent].append(item)

        self.domain_instruction = cfg.prompt.domain_instruction
        self.domain_prompt = cfg.prompt.domain_prompt
        self.domain_prompt_with_answer = cfg.prompt.domain_prompt_with_answer

        self.intent_instruction = cfg.prompt.intent_instruction
        self.intent_prompt = cfg.prompt.intent_prompt
        self.intent_prompt_with_answer = cfg.prompt.intent_prompt_with_answer

        self.slot_instruction = cfg.prompt.slot_instruction
        self.slot_prompt = cfg.prompt.slot_prompt
        self.slot_prompt_with_answer = cfg.prompt.slot_prompt_with_answer

        self.k_per_intent = cfg.prompt.k_per_intent
        self.k_per_slot = cfg.prompt.k_per_slot
        self.max_examples = cfg.prompt.max_examples

        self.__logger = logging.getLogger(self.__class__.__name__)
        experiment = mlflow.set_experiment(f"eval_single_turn_dst")
        run = mlflow.start_run(run_name=self.cfg.main.run_name)

        # Get experiment details
        self.__logger.info(f"Experiment_id: {experiment.experiment_id}")
        self.__logger.info(f"Experiment: {experiment.name}")
        self.__logger.info(f"Run: {run.info.run_name}")
        self.__logger.info(f"Artifact Location: {experiment.artifact_location}")

    def evaluate(self, data_start_index: int = 0, data_end_index: int = -1) -> EvalDSTOutput:
        eval_out = []

        num_correct_slots = 0
        num_gold_slots = 0
        num_predicted_slots = 0
        num_hallucinated_slots = 0

        y_domain_true = []
        y_domain_pred = []

        y_intent_true = []
        y_intent_pred = []

        slot_accuracy = []

        num_correct_dst = 0

        data = self.data[:data_end_index]

        mlflow.log_param('model', self.llm.model_name)

        mlflow.log_param('num_data_points', len(data))
        mlflow.log_param('data_path', self.cfg.data.data_path)

        if self.intent_prompt_with_answer and self.slot_prompt_with_answer\
                and self.intent_example_path and self.slot_example_path:
            mlflow.log_param('task', 'few-shot single-turn DST')
        else:
            mlflow.log_param('task', 'zero-shot single-turn DST')

        self.__logger.info("Starting evaluation")

        #### Domain classification
        predicted_domains = ["none" for turn in data]

        self.domain_dict_inv = self._get_domain_dict_inv()
        self.domain_options = [k for k in self.intent_dict]

        if self.domain_instruction and self.domain_prompt:
            mlflow.log_param('domain_adapter', self.llm.domain_model_name)

            predicted_domains = []

            few_shot_domain_data = []
            if self.domain_prompt_with_answer and self.intent_example_path:
                few_shot_domain_data = self._generate_few_shot_domain_data()

            filled_prompts = []
            for turn in tqdm(data, desc="Preparing data for domain classification"):
                filled_prompt = self.fill_prompt(turn=turn, task="domain", few_shot_data=few_shot_domain_data)
                filled_prompts.append(filled_prompt)

            bs = 64
            prompts_batches = util.batch(data=filled_prompts, bs=bs)

            responses = []
            for prompts in tqdm(prompts_batches, total=len(filled_prompts) // bs, desc="Generating responses"):
                outputs = self.llm.run_domain(prompts=prompts)
                responses.extend(outputs)

            for turn, filled_prompt, response in zip(tqdm(data, desc="Evaluating responses"),
                                                     filled_prompts, responses):
                predicted_domain = "none"

                if "," in response: response = response.split(",")[0]
                if "." in response: response = response.split(".")[0]
                response = response.lower().strip()
                response = re.sub(r'[^-\w\s\(\)]', '', response)
                if response in self.domain_options: predicted_domain = response
                predicted_domains.append(predicted_domain)

        mlflow.log_param('intent_adapter', self.llm.intent_model_name)
        mlflow.log_param('slot_adapter', self.llm.slot_model_name)

        #### Intent classification
        predicted_intents = []

        filled_prompts = []
        for i, turn in enumerate(tqdm(data, desc="Preparing data for intent classification")):
            intent_options = self._get_intent_options(predicted_domains[i])

            few_shot_intent_data = []
            if self.intent_prompt_with_answer and self.intent_example_path:
                few_shot_intent_data = self._generate_few_shot_intent_data()

            filled_prompt = self.fill_prompt(turn=turn, task="intent", few_shot_data=few_shot_intent_data,
                                             intent_options=intent_options)
            filled_prompts.append(filled_prompt)

        bs = 64
        prompts_batches = util.batch(data=filled_prompts, bs=bs)

        responses = []
        for prompts in tqdm(prompts_batches, total=len(filled_prompts) // bs, desc="Generating responses"):
            outputs = self.llm.run_intent(prompts=prompts)
            responses.extend(outputs)

        for i, (turn, filled_prompt, response) in enumerate(zip(tqdm(data, desc="Evaluating responses"),
                                                                filled_prompts, responses)):
            predicted_intent = "none"

            if "," in response: response = response.split(",")[0]
            if "." in response: response = response.split(".")[0]
            response = response.lower().strip()
            response = re.sub(r'[^-\w\s\(\)]', '', response)

            if response in self.intent_dict_inv:
                predicted_intent = self.intent_dict_inv[response]
            if predicted_intent in self.domain_dict_inv:
                predicted_domains[i] = self.domain_dict_inv[predicted_intent]
            predicted_intents.append(predicted_intent)

        y_domain_true = [turn.domain for turn in data]
        y_domain_pred = predicted_domains

        y_intent_true = [turn.intent for turn in data]
        y_intent_pred = predicted_intents

        #### Slot filling

        filled_prompts = []
        for i, turn in enumerate(tqdm(data, desc="Preparing data for slot filling")):
            # get possible slots to prompt given the intent in this turn
            candidate_slots = {}
            if predicted_intents[i] in self.slot_dict: candidate_slots.update(self.slot_dict[predicted_intents[i]])
            if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

            few_shot_slot_data = []
            if self.slot_prompt_with_answer and self.slot_example_path:
                few_shot_slot_data = self._generate_few_shot_slot_data(predicted_intents[i])
                if predicted_intents[i] == "none":
                    few_shot_slot_data = random.choices(few_shot_slot_data, k=10)

            filled_prompt = self.fill_prompt(turn=turn, task="slot", few_shot_data=few_shot_slot_data,
                                             slots=candidate_slots)
            filled_prompts.append(filled_prompt)

        bs = 64
        prompts_batches = util.batch(data=filled_prompts, bs=bs)

        responses = []
        for prompts in tqdm(prompts_batches, total=len(filled_prompts) // bs, desc="Generating responses"):
            outputs = self.llm.run_slot(prompts=prompts)
            responses.extend(outputs)

        for i, (turn, filled_prompt, response) in enumerate(
                zip(tqdm(data, desc="Evaluating responses"), filled_prompts, responses)):

            num_gold_slots_in_turn = 0
            num_predicted_slots_in_turn = 0
            num_correct_slots_in_turn = 0

            # get ground truth slots
            expected_slots = {}
            for s in turn.slots:
                expected_slots[s.slot_name] = s.slot_values
                num_gold_slots_in_turn += 1
                num_gold_slots += 1

            # get possible slots to prompt given the intent in this turn
            candidate_slots = {}
            if predicted_intents[i] in self.slot_dict: candidate_slots.update(self.slot_dict[predicted_intents[i]])
            if 'all' in self.slot_dict: candidate_slots.update(self.slot_dict['all'])

            if not response.endswith('}'): response += '}'
            response = response.lower().strip()

            predicted_slots = {}
            for slot_name, slot_desc in candidate_slots.items():       # find slot values in the response
                if slot_name in expected_slots:
                    gold_slot_values = expected_slots[slot_name]
                else:
                    gold_slot_values = []

                slot_pattern = re.compile(slot_name.lower() + r": (.+?)[,\.\}]")
                slot_value = re.search(slot_pattern, response)
                if slot_value:
                    slot_value = slot_value.group(1)

                    slot_value = slot_value.strip()
                    if slot_value.startswith("\"") or slot_value.startswith("'"): slot_value = slot_value[1:]
                    if slot_value.endswith("\"") or slot_value.endswith("'"): slot_value = slot_value[:-1]
                    slot_value = slot_value.strip()

                    if slot_value != "none" and \
                            slot_value != "null" and \
                            slot_value != "n/a" and \
                            slot_value != "unspecified":                    # the answer span is predicted
                        predicted_slots[slot_name] = slot_value
                        num_predicted_slots += 1
                        num_predicted_slots_in_turn += 1

                        if self.check_response_span(
                                expected=gold_slot_values,
                                response=slot_value
                        ):                                                  # the answer span is correct
                            num_correct_slots += 1
                            num_correct_slots_in_turn += 1
                        else:
                            if slot_value.lower() not in turn.utterance:    # the answer span is NOT in the utterance
                                # --> hallucinate
                                num_hallucinated_slots += 1

            sorted_expected_slots = dict(sorted(expected_slots.items()))
            sorted_predicted_slots = dict(sorted(predicted_slots.items()))

            expected_slot_values = "; ".join([f"{k}: {v}" for k, v in sorted_expected_slots.items()])
            predicted_slot_values = "; ".join([f"{k}: {v}" for k, v in sorted_predicted_slots.items()])

            expected_dst = "; ".join([y_domain_true[i], y_intent_true[i], expected_slot_values])
            predicted_dst = "; ".join([y_domain_pred[i], y_intent_pred[i], predicted_slot_values])

            if num_gold_slots_in_turn > 0:
                turn_slot_accuracy = num_correct_slots_in_turn / num_gold_slots_in_turn
                slot_accuracy.append(turn_slot_accuracy)
            else:
                turn_slot_accuracy = (num_predicted_slots_in_turn == num_gold_slots_in_turn)

            if turn_slot_accuracy == 1 and predicted_domains[i] == y_domain_true[i] and \
                    predicted_intents[i] == y_intent_true[i]:
                num_correct_dst += 1

            eval_out.append(
                Result(
                    utterance=turn.utterance,
                    filled_prompt=filled_prompt,
                    expected=expected_dst,
                    predicted=predicted_dst,
                    response=response
                )
            )

        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

        domain_accuracy = accuracy_score(y_domain_true, y_domain_pred)
        domain_precision = precision_score(y_domain_true, y_domain_pred, average='macro',
                                           labels=list(set(y_domain_true)))
        domain_recall = recall_score(y_domain_true, y_domain_pred, average='macro',
                                     labels=list(set(y_domain_true)))
        domain_f1 = f1_score(y_domain_true, y_domain_pred, average='macro',
                             labels=list(set(y_domain_true)))

        intent_accuracy = accuracy_score(y_intent_true, y_intent_pred)
        intent_precision = precision_score(y_intent_true, y_intent_pred, average='macro',
                                           labels=list(set(y_intent_true)))
        intent_recall = recall_score(y_intent_true, y_intent_pred, average='macro',
                                     labels=list(set(y_intent_true)))
        intent_f1 = f1_score(y_intent_true, y_intent_pred, average='macro',
                             labels=list(set(y_intent_true)))

        try:
            avg_slot_accuracy = np.average(slot_accuracy)
            slot_precision = num_correct_slots / num_predicted_slots
            slot_recall = num_correct_slots / num_gold_slots
            slot_f1 = (2 * slot_precision * slot_recall) / (slot_precision + slot_recall)
            slot_hallucinate = num_hallucinated_slots / num_predicted_slots
        except ZeroDivisionError:
            avg_slot_accuracy = 0
            slot_precision = 0
            slot_recall = 0
            slot_f1 = 0
            slot_hallucinate = 0

        exact_match_accuracy = num_correct_dst / len(data)

        output = EvalDSTOutput(
            model_name=self.llm.model_name,
            time=datetime.now().strftime("%d-%m-%Y-%H-%M-%S"),
            domain_instruction=self.domain_instruction,
            domain_prompt=self.domain_prompt,
            intent_instruction=self.intent_instruction,
            intent_prompt=self.intent_prompt,
            slot_instruction=self.slot_instruction,
            slot_prompt=self.slot_prompt,
            domain_accuracy=domain_accuracy,
            domain_precision=domain_precision,
            domain_recall=domain_recall,
            domain_f1=domain_f1,
            intent_accuracy=intent_accuracy,
            intent_precision=intent_precision,
            intent_recall=intent_recall,
            intent_f1=intent_f1,
            avg_slot_accuracy=avg_slot_accuracy,
            slot_precision=slot_precision,
            slot_recall=slot_recall,
            slot_f1=slot_f1,
            slot_hallucinate=slot_hallucinate,
            result=eval_out
        )

        self.__logger.info("Finishing evaluation")
        self.__logger.info(f"Total count: {len(data)}")
        self.__logger.info(f"Correct count: {num_correct_dst}")
        self.__logger.info(f"Total slots: {num_gold_slots}")
        self.__logger.info(f"Correct slots: {num_correct_slots}")
        self.__logger.info(f"Exact Match Accuracy: {exact_match_accuracy}")
        self.__logger.info(f"Domain Accuracy: {domain_accuracy}")
        self.__logger.info(f"Domain Precision: {domain_precision}")
        self.__logger.info(f"Domain Recall: {domain_recall}")
        self.__logger.info(f"Domain F1: {domain_f1}")
        self.__logger.info(f"Intent Accuracy: {intent_accuracy}")
        self.__logger.info(f"Intent Precision: {intent_precision}")
        self.__logger.info(f"Intent Recall: {intent_recall}")
        self.__logger.info(f"Intent F1: {intent_f1}")
        self.__logger.info(f"Avg Slot Accuracy: {avg_slot_accuracy}")
        self.__logger.info(f"Slot Precision: {slot_precision}")
        self.__logger.info(f"Slot Recall: {slot_recall}")
        self.__logger.info(f"Slot F1: {slot_f1}")
        self.__logger.info(f"Slot Hallucinate: {slot_hallucinate}")

        file_path = f"evaluation/logs/eval_output_{self.cfg.main.run_name}.json"

        self.save(file_path=file_path, output=output.dict())
        self.__logger.info(f"Saved evaluation results at {file_path}")

        mlflow.log_param('total_slots', num_gold_slots)
        mlflow.log_param('correct_slots', num_correct_slots)
        mlflow.log_metrics(
            {
                "domain accuracy": domain_accuracy,
                "domain precision": domain_precision,
                "domain recall": domain_recall,
                "domain f1": domain_f1,
                "intent accuracy": intent_accuracy,
                "intent precision": intent_precision,
                "intent recall": intent_recall,
                "intent f1": intent_f1,
                "avg slot accuracy": avg_slot_accuracy,
                "slot precision": slot_precision,
                "slot recall": slot_recall,
                "slot f1": slot_f1,
                "slot hallucinate": slot_hallucinate,
                "exact match accuracy": exact_match_accuracy
            }
        )
        mlflow.log_artifact(file_path)
        mlflow.log_artifact(self.intent_desc_path)
        mlflow.log_artifact(self.slot_desc_path)

        config_path = f"evaluation/logs/config_{self.cfg.main.run_name}.yaml"
        OmegaConf.save(self.cfg, config_path)
        mlflow.log_artifact(config_path)

        # Save the confusion matrix
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay
        disp = ConfusionMatrixDisplay.from_predictions(y_domain_true, y_domain_pred,
                                                       labels=list(set(y_domain_true)),
                                                       cmap=plt.cm.Blues,
                                                       xticks_rotation='vertical',
                                                       normalize="true")

        png_path = f"evaluation/logs/confusion_{self.cfg.main.run_name}.png"
        fig = disp.figure_
        if len(list(set(y_domain_true))) > 50:
            fig.set_figwidth(25)
            fig.set_figheight(25)
        elif len(list(set(y_domain_true))) > 30:
            fig.set_figwidth(20)
            fig.set_figheight(20)
        elif len(list(set(y_domain_true))) > 10:
            fig.set_figwidth(15)
            fig.set_figheight(15)
        else:
            fig.set_figwidth(10)
            fig.set_figheight(10)

        fig.savefig(png_path)
        mlflow.log_artifact(png_path)

        disp = ConfusionMatrixDisplay.from_predictions(y_intent_true, y_intent_pred,
                                                       labels=list(set(y_intent_true)),
                                                       cmap=plt.cm.Blues,
                                                       xticks_rotation='vertical',
                                                       normalize="true")

        png_path = f"evaluation/logs/confusion_{self.cfg.main.run_name}.png"
        fig = disp.figure_
        if len(list(set(y_intent_true))) > 50:
            fig.set_figwidth(25)
            fig.set_figheight(25)
        elif len(list(set(y_intent_true))) > 30:
            fig.set_figwidth(20)
            fig.set_figheight(20)
        elif len(list(set(y_intent_true))) > 10:
            fig.set_figwidth(15)
            fig.set_figheight(15)
        else:
            fig.set_figwidth(10)
            fig.set_figheight(10)

        fig.savefig(png_path)
        mlflow.log_artifact(png_path)

        return output


    def fill_prompt(self, turn: SlotFillingInstance, task: str, few_shot_data: list = [], intent_options: list = [], slots: dict = {}) -> str:
        if task == "domain":
            few_shot_instances = [self.domain_prompt_with_answer.format(**fs.dict()) for fs in few_shot_data]
            if len(few_shot_instances) > 10:  # limit few-shot examples
                few_shot_instances = random.choices(few_shot_instances, k=10)

            prompt = self.domain_instruction.format(domain_options=" - " + " \n - ".join(self.domain_options) + "\n")
            if few_shot_instances: prompt += '\n' + '\n'.join(few_shot_instances) + '\n'
            prompt += self.domain_prompt.format(**turn.dict())
            return prompt

        elif task == "intent":
            few_shot_instances = [self.intent_prompt_with_answer.format(**fs.dict()) for fs in few_shot_data]
            if len(few_shot_instances) > 10:  # limit few-shot examples
                few_shot_instances = random.choices(few_shot_instances, k=10)

            prompt = self.intent_instruction.format(intent_options=" - " + " \n - ".join(intent_options) + "\n")
            if few_shot_instances: prompt += '\n'.join(few_shot_instances) + '\n'
            prompt += self.intent_prompt.format(**turn.dict())
            return prompt

        elif task == "slot":
            few_shot_slots = [{f"{slot.slot_name}": f"{slot.slot_values[0]}" for slot in fs.slots} for fs in
                              few_shot_data]

            few_shot_instances = [self.slot_prompt_with_answer.format(
                utterance=fs.utterance,
                slots=", ".join([f"{slot}: {few_shot_slots[i][slot]}"
                                 if slot in few_shot_slots[i]
                                 else f"{slot}: none"
                                 for slot in slots])
            ) for i, fs in enumerate(few_shot_data)]

            prompt = self.slot_instruction.format(slot_dict="\n\t" + "\n\t".join([f"{k}: {v}," for k, v in slots.items()]) + "\n")
            if few_shot_instances: prompt += "\n".join(few_shot_instances)
            prompt += self.slot_prompt.format(**turn.dict())
            return prompt

    def _generate_few_shot_domain_data(self):
        few_shot_data = []
        for key in self.examples_per_domain:
            for item in random.sample(self.examples_per_domain[key], k=self.k_per_intent):
                few_shot_data.append(item)

        if len(few_shot_data) > self.max_examples:
            few_shot_data = random.sample(few_shot_data, k=self.max_examples)

        random.shuffle(few_shot_data)
        return few_shot_data

    def _generate_few_shot_intent_data(self):
        few_shot_data = []
        for key in self.examples_per_intent:
            for item in random.sample(self.examples_per_intent[key], k=self.k_per_intent):
                few_shot_data.append(item)

        if len(few_shot_data) > self.max_examples:
            few_shot_data = random.sample(few_shot_data, k=self.max_examples)

        random.shuffle(few_shot_data)
        return few_shot_data

    def _generate_few_shot_slot_data(self, intent: str):
        few_shot_data = []
        few_shot_examples = EvalDataSF.parse_file(self.slot_example_path).data
        if intent != "none":
            few_shot_examples = [turn for turn in few_shot_examples if turn.intent == intent]  # filter data only with a certain intent

        samples_per_slot_name = {}
        for idx, sample in enumerate(few_shot_examples):
            for slot in sample.slots:
                if slot.slot_name not in samples_per_slot_name: samples_per_slot_name[slot.slot_name] = []
                samples_per_slot_name[slot.slot_name].append(idx)

        counts_per_slot_name = {}
        for slot_name in samples_per_slot_name: counts_per_slot_name[slot_name] = self.k_per_slot

        for slot_name in samples_per_slot_name:
            for count in range(counts_per_slot_name[slot_name]):
                random_idx = random.choice(samples_per_slot_name[slot_name])
                random_sample = few_shot_examples[random_idx]
                few_shot_data.append(random_sample)

                for slot in random_sample.slots:
                    counts_per_slot_name[slot.slot_name] -= 1

        return few_shot_data
