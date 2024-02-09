import json

from abc import abstractmethod, ABC
from typing import List, Dict, Union

from evaluation.dtos.dto import \
    IntentClassificationInstance, \
    SlotFillingInstance


class BuildEvalData(ABC):

    @abstractmethod
    def get_data(self, paths: List[str]):
        ...

    @abstractmethod
    def build_ic_data(self, output_path: str) -> List[IntentClassificationInstance]:
        ...

    @abstractmethod
    def build_sf_data(self, output_path: str) -> List[SlotFillingInstance]:
        ...

    @abstractmethod
    def build_few_shot_ic_data(self, output_path: str) -> List[IntentClassificationInstance]:
        ...

    @abstractmethod
    def build_few_shot_sf_data(self, output_path: str) -> List[SlotFillingInstance]:
        ...

    @staticmethod
    def save_as_json(data: Dict, output_path: str) -> None:
        with open(output_path, "w+") as f:
            json.dump(data, f, indent=4)
