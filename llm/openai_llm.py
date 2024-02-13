import os
from openai import OpenAI

from llm.llm import LLM
from typing import Union, List

class OpenAILLM(LLM):
    def __init__(
        self,
        model: str = "gpt-3.5-turbo-instruct"
    ):
        self.model_name = model
        self.client = OpenAI()

        self.domain_model_name = self.model_name
        self.intent_model_name = self.model_name
        self.slot_model_name = self.model_name

    def run(
        self,
        prompts: Union[str, List[str]],
        max_new_tokens: int = 10,
        split_lines: bool = False,
        temperature: float = 0,
        frequency_penalty: float = 2.0,
        presence_penalty: float = 2.0
    ) -> str:
        assert 'OPENAI_API_KEY' in os.environ.keys(), \
            "Please set your OPENAI_API_KEY!"
        responses = self.client.completions.create(
            model=self.model_name,
            prompt=prompts,
            temperature=temperature,
            max_tokens=max_new_tokens,
            top_p=1,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            stop=["\\n"]
        ).choices

        responses = [response.text.strip() for response in responses]
        if split_lines:
            responses_post = []
            for response in responses:
                if response:
                    responses_post.append(response.splitlines()[0])
                else:
                    responses_post.append(response)

        return responses

    def run_domain(self, prompts: Union[str, List[str]]) -> str:
        return self.run(prompts, max_new_tokens=10, split_lines=True)

    def run_intent(self, prompts: Union[str, List[str]]) -> str:
        return self.run(prompts, max_new_tokens=10, split_lines=True)

    def run_slot(self, prompts: Union[str, List[str]]) -> str:
        return self.run(prompts, max_new_tokens=100, split_lines=False)
