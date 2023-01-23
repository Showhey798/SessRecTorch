from typing import Optional, Dict
#import tensorflow as tf
import os

from logging import Logger, getLogger, StreamHandler, DEBUG, Formatter

from csv import writer

import mlflow
from omegaconf import DictConfig, ListConfig

def create_logger(
    name:Optional[str]=None
):
    logger = getLogger(name)
    logger.setLevel(DEBUG)
    logger.propagate = False
    
    ch = StreamHandler()
    ch.setLevel(DEBUG)
    ch.setFormatter(Formatter("%(asctime)s - %(filename)s - %(message)s", "%Y-%m-%d %H:%M:%S"))
    logger.addHandler(ch)
    return logger

class ResultLogger(object):
    
    def __init__(self, file_path, k):
        self.file_path = os.path.join(file_path, "result@%d.csv"%k)
        self.k = k
        if not os.path.exists(self.file_path):
            self.write_line("Model", "ClickType", "Hit_at_%d"%k, "NDCG_at_%d"%k)
        
    def write_line(self, model, clicktype, hit, ndcg):
        tocsv_list = [model, clicktype, hit, ndcg]
        with open(self.file_path, "a", newline='') as f:
            writer_obj = writer(f)
            writer_obj.writerow(tocsv_list)
            f.close()

def log_params_from_omegaconf_dict(params):
    for param_name, element in params.items():
        _explore_recursive(param_name, element)

def _explore_recursive(parent_name, element):
    if isinstance(element, DictConfig):
        for k, v in element.items():
            if isinstance(v, DictConfig) or isinstance(v, ListConfig):
                _explore_recursive(f'{parent_name}.{k}', v)
            else:
                mlflow.log_param(f'{parent_name}.{k}', v)
    elif isinstance(element, ListConfig):
        for i, v in enumerate(element):
            mlflow.log_param(f'{parent_name}.{i}', v)