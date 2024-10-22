import torch, copy
from torch import nn
import numpy as np
from typing import List, Union, Dict

from .utils import stack_all, object_to_list, collate_all_list, detach_all
from .utils import get_dic_emb_dims, get_loss_by_name, map_encoders 
from .fusion_module import FusionModuleMissing, POOL_FUNC_NAMES
from .core_fusion import _BaseViewsLightning
from .missing_utils import possible_missing_mask, augment_all_missing, augment_random_missing

class MVFusionMissing(_BaseViewsLightning):
    """
        Only for Point-wise prediction (pixel-based)
        it is based on three modules: encoders, aggregation, prediction_head
            support list and dictionary of encoders -- but transform to dict (ideally it should be always a dict)
                In this case, view_names has to be defined
            support list and dictionary of emb dims -- but transform to dict
            view_encoders has to have get_output_size() method
    """
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]], 
                 fusion_module: nn.Module,
                 prediction_head: nn.Module,
                 loss_args: dict ={},
                 view_names: List[str] = [], 
                 **kwargs,
                 ):
        super(MVFusionMissing, self).__init__(**kwargs)
        if len(view_encoders) == 0:
            raise Exception("you have to give a encoder models (nn.Module), currently view_encoders=[] or {}")
        if type(prediction_head) == type(None):
            raise Exception("you need to define a prediction_head")
        if type(fusion_module) == type(None):
            raise Exception("you need to define a fusion_module")
        if len(loss_args) == 0:
            loss_args = {"name": "mse"}
        self.save_hyperparameters(ignore=['view_encoders','prediction_head', 'fusion_module'])

        view_encoders = map_encoders(view_encoders, view_names=view_names)
        self.views_encoder = nn.ModuleDict(view_encoders)
        self.fusion_module = fusion_module
        self.prediction_head = prediction_head

        self.view_names = list(self.views_encoder.keys())
        self.N_views = len(self.view_names)
        self.criteria = loss_args["function"] if "function" in loss_args else get_loss_by_name(**self.hparams_initial.loss_args)
        self.missing_as_aug = False
            
    def set_missing_info(self, aug_status, name:str="impute", where:str ="", value_fill=None, missing_random: bool=False, random_perc = 0,**kwargs):
        #set the status of the missing as augmentation technique used during training
        self.missing_as_aug = aug_status
        if name =="impute": #for case of impute
            where = "input" if where == "" else where #default value: input
            value_fill = 0.0 if type(value_fill) == type(None) else value_fill 
        elif name == "adapt": #for case of adapt
            where = "feature" if where =="" else where #default value: input
            value_fill = torch.nan if type(value_fill) == type(None) else value_fill 
        elif name == "ignore": #completly ignore/drop missing data
            pass
        self.missing_method = {"name": name, "where": where, "value_fill": value_fill}
        if self.missing_as_aug:
            self.all_missing_views = augment_all_missing(self.view_names)
            print("The missing as augmentation was set on with an augmentation of ", possible_missing_mask(len(self.view_names)),"and following missing_method =",self.missing_method) #len(self.views_encoder.keys())

        #baseline setting
        self.missing_random = missing_random
        self.random_perc = random_perc
        
    def forward_encoders(self,
            views: Dict[str, torch.Tensor],
            inference_views: list = [],
            missing_method: dict = {},
            ) -> Dict[str, torch.Tensor]:
        if type(views) == list:
            raise Exception("Please feed forward function with dictionary data {view_name_str: torch.Tensor} instead of list")
        inference_views = self.view_names if len(inference_views) == 0 else inference_views

        zs_views = {}
        for v_name in self.view_names:
            forward_f = True  #Flag to activate view-forward 
            if v_name in inference_views and v_name in views:
                data_forward = views[v_name]

            else: 
                if missing_method.get("where") == "input": #for ablation -- fill when view not in testing forward or view is missing
                    data_forward = torch.ones_like(views[v_name])*missing_method["value_fill"] 

                elif missing_method.get("where") == "feature": #for ablation -- avoid forward and fill at feature
                    forward_f = False
                    value_fill = torch.nan if missing_method["value_fill"] == "nan" else missing_method["value_fill"]
                    zs_views[v_name] = torch.ones(self.views_encoder[v_name].get_output_size(), 
                                                  device=self.device).repeat(list(views.values())[0].shape[0], 1)*value_fill
                
                elif missing_method.get("name") == "ignore":  #default
                    forward_f=False
                    
                else:
                    raise Exception("Inference with few number of views (missing) but no missing method *where* was indicated in the arguments")

            if forward_f:
                zs_views[v_name] = self.views_encoder[v_name](data_forward)
        return {"views:rep": zs_views}

    def forward(self,
            views: Dict[str, torch.Tensor],
            intermediate:bool = False,
            out_norm:bool=False,
            inference_views: list = [],
            missing_method: dict = {}, 
            forward_from_representation: bool = False, 
            ) -> Dict[str, torch.Tensor]:
        #encoders        
        if forward_from_representation:
            out_zs_views = {"views:rep": (views if "views:rep" not in views else views["views:rep"])}
        else:
            out_zs_views = self.forward_encoders(views, inference_views=inference_views, missing_method=missing_method) 
        
        #merge function
        if len(inference_views) != 0 and missing_method.get("name") == "ignore": 
            views_data = [ out_zs_views["views:rep"][v] for v in self.view_names if v in inference_views] 
        else: #adapt, impute or others
            views_data = [ out_zs_views["views:rep"][v] for v in self.view_names] # this ensures that the same views are passed for training
        views_available_ohv = torch.ones(self.N_views) if len(inference_views) == 0 else torch.Tensor([1 if v in inference_views else 0 for v in self.view_names])
        out_z_e = self.fusion_module(views_data, views_available=views_available_ohv.bool())
        
        #prediction head
        out_y = self.prediction_head(out_z_e["joint_rep"], inference_views) #inference_views only for Presto
        return_dic = {"prediction": self.apply_softmax(out_y) if out_norm else out_y }
        if intermediate:
            return_dic["last_layer"] = out_y
            return dict( **return_dic, **out_z_e) #**out_zs_views
        else:
            return return_dic

    def prepare_batch(self, batch: dict, return_target=True) -> list:
        views_data, views_target = batch["views"], batch["target"]

        if type(views_data) == list:
            if "view_names" in batch:
                if len(batch["view_names"]) != 0:
                    views_to_match = batch["view_names"]
            else:
                views_to_match = self.view_names #assuming a order list with view names based on the construction of the class
            views_dict = {views_to_match[i]: value for i, value in enumerate(views_data) }
        elif type(views_data) == dict:
            views_dict = views_data
        else:
            raise Exception("views in batch should be a List or Dict")
        
        if return_target:
            if type(self.criteria) == torch.nn.CrossEntropyLoss:
                views_target = torch.squeeze(views_target)
            else:
                views_target = views_target.to(torch.float32)
        else:
            views_target = None
        return views_dict, views_target

    def loss_batch(self, batch: dict) -> dict:
        views_dict, views_target = self.prepare_batch(batch)
        if self.missing_as_aug and self.training:
            if self.missing_random: #a single random augmentation instance -- for baseline and ablation
                missing_case = augment_random_missing(self.view_names, perc=self.random_perc)
                out_dic = self(views_dict, inference_views=missing_case, missing_method=self.missing_method)
                views_targets = views_target

            else:
                zs_views = self.forward_encoders(views_dict)["views:rep"]

                views_targets = []
                for i, inference_views in enumerate(self.all_missing_views):
                    if self.missing_method.get("name") == "ignore": #default
                        aux_zs_views = {k: v for k,v in zs_views.items()} 
                    else: 
                        value_fill = torch.nan if self.missing_method["value_fill"] == "nan" else self.missing_method["value_fill"] 
                        aux_zs_views = {k: (v if k in inference_views else torch.ones_like(v)*value_fill) for k, v in zs_views.items()}
                        
                    out_dic_ = self(aux_zs_views, inference_views=inference_views, missing_method=self.missing_method, forward_from_representation = True)
                    
                    if i == 0:
                        out_dic = object_to_list(out_dic_) 
                    else:
                        collate_all_list(out_dic, out_dic_)
                    views_targets.append(views_target)
                out_dic = stack_all(out_dic, data_type="torch")
                views_targets = torch.concat(views_targets, axis = 0)
        else: 
            out_dic = self(views_dict) 
            views_targets = views_target

        if self.hparams_initial.get("focus_full_view") and (len(views_target) != len(views_targets)): #for baseline
            return_dic =  {"objective": self.criteria(out_dic["prediction"][:len(views_target)], views_target) +
                    self.criteria(out_dic["prediction"][len(views_target):], views_targets[len(views_target):])}
        
        else: #default
            return_dic = {"objective": self.criteria(out_dic["prediction"], views_targets)}

        if self.missing_as_aug and self.training and not self.missing_random: #multiple missing views cases
            return_dic["fullview"] =  self.criteria(out_dic["prediction"][:len(views_target)], views_target)
            return_dic["missingview"] =  self.criteria(out_dic["prediction"][len(views_target):], views_targets[len(views_target):])
        return return_dic    
            
    def transform(self,
            loader: torch.utils.data.DataLoader,
            intermediate=True,
            out_norm=False,
            device:str="",
            args_forward:dict = {},
            perc_forward: float = 1,
            **kwargs
            ) -> dict:
        """
        function to get predictions from model  -- inference or testing

        :param loader: a dataloader that matches the structure of that used for training
        :return: transformed views

        #return numpy arrays based on dictionary
        """
        device = ("cuda" if torch.cuda.is_available() else "cpu") if device == "" else device
        device_used = torch.device(device)

        missing_forward = True 
        self.eval() 
        self.to(device_used)
        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                views_dict, _ = self.prepare_batch(batch, return_target=True)
                for view_name in views_dict:
                    views_dict[view_name] = views_dict[view_name].to(device_used)

                if perc_forward != 1 and perc_forward !=0: 
                    if np.random.rand() > perc_forward: 
                        missing_forward = False
                    
                if missing_forward:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm, **args_forward)
                else:
                    outputs_ = self(views_dict, intermediate=intermediate, out_norm=out_norm)
                missing_forward = True

                outputs_ = detach_all(outputs_)
                if batch_idx == 0:
                    outputs = object_to_list(outputs_) 
                else:
                    collate_all_list(outputs, outputs_)
        self.train()
        return stack_all(outputs)


class SVPool(MVFusionMissing):
    """
        train single-view learning models in a pool, independently between each other
        the awareness_vectors is like a positional encoding for each view
    """
    def __init__(self,
                 view_encoders: Union[List[nn.Module],Dict[str,nn.Module]],
                 view_prediction_heads: Union[List[nn.Module],Dict[str,nn.Module]],
                 loss_args: dict ={},
                 view_names: List[str] = [],
                 awareness_vectors: dict = {},
                 awareness_merge: str = "",
                 **kwargs,
                 ):
        super(SVPool, self).__init__(view_encoders, nn.Identity(), nn.Identity(),
            loss_args=loss_args, view_names=view_names, **kwargs)    
        
        self.view_prediction_heads = nn.ModuleDict(view_prediction_heads)
        self.prediction_aggregation = FusionModuleMissing([v.get_output_size() for v in self.view_prediction_heads.values()], mode="avg_ignore") #average by ignoring missing predictions
        self.awareness_vectors = nn.ParameterDict(awareness_vectors)
        self.awareness_merge = awareness_merge
        
    def forward(self, views: Dict[str, torch.Tensor], out_norm=False, inference_views: list = [], **kwargs):
        out_zs_views = self.forward_encoders(views)
                
        out_y_zs = {}
        for v_name in out_zs_views["views:rep"]:
            if len(self.awareness_vectors) != 0:
                if self.awareness_merge == "sum":
                    out_zs_views["views:rep"][v_name] += self.awareness_vectors[v_name]
                elif self.awareness_merge == "prod":
                    out_zs_views["views:rep"][v_name] *= self.awareness_vectors[v_name]
                elif self.awareness_merge == "concat":
                    out_zs_views["views:rep"][v_name] = torch.concat([out_zs_views["views:rep"][v_name],self.awareness_vectors[v_name][None,:].expand(out_zs_views["views:rep"][v_name].shape[0],-1)],dim=-1)
                else:
                    raise Exception("Incorrect merge in the awareness vector")
                
            if len(inference_views) != 0 :
                if v_name in inference_views: 
                    out_y = self.view_prediction_heads[v_name]( out_zs_views["views:rep"][v_name])
                else:
                    out_y  = torch.ones(len(out_zs_views["views:rep"][v_name]), self.view_prediction_heads[v_name].get_output_size(), device=self.device)*torch.nan
            else:
                out_y = self.view_prediction_heads[v_name]( out_zs_views["views:rep"][v_name])
            out_y_zs[v_name] = self.apply_softmax(out_y) if out_norm else out_y
        out_y_zs["aggregated"] = self.prediction_aggregation(list(out_y_zs.values()), views_available=[])["joint_rep"]
        
        return {"views:prediction": out_y_zs}

    def loss_batch(self, batch: dict):
        views_dict, views_target = self.prepare_batch(batch)
        out_dic = self(views_dict)
        yi_xi = out_dic["views:prediction"]

        loss_dic = { }
        loss_aux = 0
        for v_name in self.view_names: 
            loss_dic["loss"+v_name] = self.criteria(yi_xi[v_name], views_target)
            loss_aux += loss_dic["loss"+v_name]
        return {"objective": loss_aux, **loss_dic}

    def get_sv_models(self):
        return self.view_prediction_heads
