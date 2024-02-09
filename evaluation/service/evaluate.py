import re
import json

from abc import ABC, abstractmethod
from typing import List, Union, Dict
from rapidfuzz import fuzz

from evaluation.dtos.dto import EvalOutput, SlotFillingInstance, \
    IntentClassificationInstance
from evaluation.dtos.config import Config
from llm.llm import LLM


class EvaluateLLM(ABC):
    def __init__(self, prompt: str, llm: LLM, data_path: str, cfg: Config):
        # self.data = json.load(open(data_path, "r"))
        self.prompt = prompt
        self.llm = llm
        self.cfg = cfg

    @abstractmethod
    def evaluate(self) -> EvalOutput:
        ...

    @abstractmethod
    def fill_prompt(self, turn: Union[SlotFillingInstance, IntentClassificationInstance]) -> str:
        ...

    def get_response(self, prompt: str) -> str:
        return self.llm.run(prompt=prompt)

    def check_response(
        self, expected: str, response: str, expected_options: List[str] = None
        ) -> bool:
        response = response.lower()
        expected = expected.lower()

        response = re.sub(r'[^\w\s\(\)]', '', response)
        expected = re.sub(r'[^\w\s\(\)]', '', expected)

        if expected_options:
            expected_options = [opt.lower() for opt in expected_options]

            expected_option_numbers = [re.search("(\(\d+\))", opt) for opt in expected_options]
            expected_options = [re.sub(r'\(\d+\)', '', opt).strip() for opt in expected_options]

            predicted_count = 0
            predicted_number_count = 0
            for i, option in enumerate(expected_options):
                if option in response or response in option:
                    predicted_count += 1

                if expected_option_numbers[i]:
                    if expected_option_numbers[i].group(1) in response:
                        predicted_number_count += 1

            if predicted_count == 1 or predicted_number_count == 1:     # Found only one option in the answer
                expected_number = re.search("(\(\d+\))", expected)
                response_number = re.search("(\(\d+\))", response)

                expected = re.sub(r'\(\d+\)', '', expected).strip()
                response = re.sub(r'\(\d+\)', '', response).strip()

                if expected in response:
                    return True
                elif expected_number and response_number:
                    if expected_number.group(1) == response_number.group(1):
                        return True

                return False

            else:   # Found several options in the answer
                return False
        else:
            expected_number = re.search("(\(\d+\))", expected)
            response_number = re.search("(\(\d+\))", response)

            expected = re.sub(r'\(\d+\)', '', expected).strip()
            response = re.sub(r'\(\d+\)', '', response).strip()

            if expected in response:
                return True
            elif expected_number and response_number:
                if expected_number.group(1) == response_number.group(1):
                    return True

            return False

    def check_response_span(self, expected: list, response: str) -> bool:
        response = response.lower()
        response = re.sub(r'[^\w\s]', '', response)  # cleaning up from punctuations

        for gold in expected:
            gold = gold.lower()
            gold = re.sub(r'[^\w\s]', '', gold)  # cleaning up from punctuations
            if response in gold or fuzz.ratio(response, gold) > 80:
                return True

        return False


    def save(self, file_path: str, output: Dict) -> None:
        with open(file_path, 'w+', encoding='utf-8') as f:
            json.dump(output, f, indent=4)
