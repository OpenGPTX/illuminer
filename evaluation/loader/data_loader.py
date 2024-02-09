from typing import Union
from pydantic import parse_file_as

from evaluation.dtos.dto import EvalDataIC, EvalDataSF


class LoadEvalData:
    def __init__(self, data_path: str) -> None:
        self.data = parse_file_as(
            path=data_path,
            type_=Union[EvalDataSF, EvalDataIC]
        )
