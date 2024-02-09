from abc import ABC, abstractmethod
from typing import Union, List


class LLM(ABC):
    model_name = None

    @abstractmethod
    def run(self, prompts: Union[str, List[str]], **args) -> str:
        ...
