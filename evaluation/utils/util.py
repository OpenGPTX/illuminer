from datetime import datetime
from pathlib import Path
from typing import Dict

def get_suffix(model_name: str) -> str:
    time_now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    suffix = model_name + '-' + time_now
    return suffix

def get_run_name(config_choices: Dict[str, str]):
    return '_'.join([Path(config_choices['prompt']).stem,
                     Path(config_choices['data']).stem,
                     Path(config_choices['model']).stem]
                    )

def batch(data, bs=1):
    l = len(data)
    for ndx in range(0, l, bs):
        yield data[ndx:min(ndx + bs, l)]
