from hydra import initialize, compose
from omegaconf import DictConfig, OmegaConf


def get_config(config_path: str = "../config", config_name: str = "config") -> DictConfig:
    with initialize(version_base=None, config_path=f'../../{config_path}'):
        cfg = compose(config_name=config_name)
        return OmegaConf.to_object(cfg)
