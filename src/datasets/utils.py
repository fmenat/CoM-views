import numpy as np

from typing import List, Union, Dict

from .views_loader import DataViews_torch

def _to_loader(data: Union[List[np.ndarray],Dict[str,np.ndarray]], batch_size=32, train=True , args_loader={}, **args_structure):
    if type(data) == dict:
        aux_str = DataViews_torch(**data, **args_structure, train=train)
    else:
        aux_str = DataViews_torch(data, **args_structure, train=train)
    return aux_str.get_torch_dataloader(batch_size = batch_size, **args_loader)