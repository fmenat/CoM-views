from torch import nn
from typing import List, Union, Dict

from .base_fusion import MVFusionMissing, SVPool
from .fusion_module import FusionModuleMissing

class FeatureFusion(MVFusionMissing):
    def __init__(self,
                 view_encoders,
                 fusion_module: nn.Module,
                 predictive_model: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 **kwargs,
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        super(FeatureFusion, self).__init__(view_encoders, fusion_module, predictive_model,
             loss_args=loss_args, view_names=view_names, **kwargs)


#------------- BASELINES ----------------
class InputFusion(MVFusionMissing):
    def __init__(self,
                 predictive_model,
                 fusion_module: dict = {},
                 loss_args: dict = {},
                 view_names: List[str] = [],
                 input_dim_to_stack: Union[List[int], Dict[str,int]] = 0,
                 **kwargs,
                 ):
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        if type(fusion_module) == dict:
            if len(fusion_module) == 0:
                fusion_module = {"mode": "concat", "adaptive":False, "emb_dims": input_dim_to_stack }
            fusion_module = FusionModuleMissing(**fusion_module)
        fake_view_encoders = []
        for v in fusion_module.emb_dims:
            aux = nn.Identity()
            aux.get_output_size = lambda : v
            fake_view_encoders.append( aux)
        super(InputFusion, self).__init__(fake_view_encoders, fusion_module, predictive_model,
            loss_args=loss_args, view_names=view_names, **kwargs)
        

class SingleViewPool(SVPool):
    pass