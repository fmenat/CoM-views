import torch
from torch import nn
import numpy as np
import itertools
from typing import List, Dict

from .nn_models import create_model
from .single.encoders import RNNet,TransformerNet
from .single.fusion_layers import LinearSum_,UniformSum_,Product_,Maximum_,Stacking_,Concatenate_, STACK_FUNC_NAMES, POOL_FUNC_NAMES

class FusionModuleMissing(nn.Module):
    def __init__(self, 
                 emb_dims: List[int], 
                 mode: str, 
                 adaptive: bool=False, 
                 features: bool=False, 
                 pos_encoding: bool = False,
                 permute_rnn: bool = False,
                 random_permute:bool = True,
                 awareness: bool = False,
                 un_awareness: bool = False,
                 **kwargs
                 ):
        super(FusionModuleMissing, self).__init__()
        self.mode = mode
        self.adaptive = adaptive
        self.pos_encoding = pos_encoding
        self.permute_rnn = permute_rnn
        self.random_permute = random_permute
        self.awareness = awareness
        self.un_awareness = un_awareness
        self.emb_dims = list(emb_dims.values()) if type(emb_dims) == dict  else emb_dims
        self.N_views = len(emb_dims)
        self.joint_dim, self.feature_pool = self.get_dim_agg()
        self.check_valid_args()
        
        if self.feature_pool:
            self.stacker_function = Stacking_()

        if self.mode in STACK_FUNC_NAMES:
            self.concater_function = Concatenate_()

        elif self.mode.split("_")[0] in ["avg","mean","uniformsum"]:
            self.pooler_function = UniformSum_(ignore = self.mode.split("_")[-1] == "ignore" )

        elif self.mode.split("_")[0] in ["sum","add","linearsum"]:
            self.pooler_function = LinearSum_(ignore = self.mode.split("_")[-1] == "ignore")

        elif self.mode.split("_")[0] in ["prod", "mul"]:
            self.pooler_function = Product_(ignore = self.mode.split("_")[-1] == "ignore")

        elif self.mode.split("_")[0] in ["max", "pool"]:
            self.pooler_function = Maximum_(ignore = self.mode.split("_")[-1] == "ignore")
        
        elif self.mode.split("_")[0] in ["rnn", "lstm", "gru"]:
            self.permu_opts_viewsfunc = lambda n_views: list(itertools.permutations(np.arange(n_views)))
            self.pooler_function = RNNet(feature_size=self.joint_dim, layer_size=self.joint_dim, unit_type=self.mode.split("_")[0], output_state=True, **kwargs)

        elif self.mode.split("_")[0] in ["transformer", "trans"]:
            self.pooler_function = TransformerNet(feature_size=self.joint_dim, layer_size=self.joint_dim, len_max_seq=self.N_views, **kwargs)

        elif self.mode.split("_")[0] in ["sampling"]:
            self.features = features
            pass 

        else:
            raise ValueError(f'Invalid value for mode: {self.mode}. Valid values: {POOL_FUNC_NAMES+STACK_FUNC_NAMES}')

        if self.adaptive:
            self.features = features
            out_probs = self.N_views
            forward_input_dim = sum(self.emb_dims) if self.mode in STACK_FUNC_NAMES else self.joint_dim
            forward_output_dim = self.joint_dim*out_probs if self.features else out_probs

            if "adaptive_args" in kwargs:
                self.attention_function = create_model(forward_input_dim, forward_output_dim, layer_size=forward_input_dim, **kwargs["adaptive_args"])
            else:
                self.attention_function = nn.Linear(forward_input_dim, forward_output_dim, bias=True)            

        if self.pos_encoding and self.feature_pool:
            self.pos_encoder = nn.Linear(self.N_views, self.joint_dim, bias=False)
            self.ohv_basis = nn.functional.one_hot(torch.arange(0,self.N_views), num_classes=self.N_views).float()
        
    def get_dim_agg(self):
        if self.adaptive or (self.mode.split("_")[0] not in STACK_FUNC_NAMES):
            fusion_dim = self.emb_dims[0]
            feature_pool = True
        else:
            fusion_dim = sum(self.emb_dims)
            feature_pool = False
        return fusion_dim, feature_pool

    def check_valid_args(self):
        if len(np.unique(self.emb_dims)) != 1:
            if self.adaptive:
                raise Exception("Cannot set adaptive=True when the number of features in embedding are not the same")
            if self.mode.split("_")[0] in POOL_FUNC_NAMES + ["sampling"]:
                raise Exception("Cannot set pooling aggregation when the number of features in embedding are not the same")


    def forward(self, views_emb: List[torch.Tensor], views_available: torch.Tensor) -> Dict[str, torch.Tensor]: 
        """
            * views_emb: list of tensors with shape (N_batch, N_dims) for each view. It can be less than N_views if some views are missing
            * views_available: tensor with shape (N_views) with 1 if view is available and 0 if not
        """
        n_views_available = views_available.sum() if len(views_available) > 0 else self.N_views
        missing_boolean = n_views_available < self.N_views 

        if self.feature_pool:
            if self.pos_encoding:
                encodings = self.pos_encoder(self.ohv_basis.to(views_emb[0].device))
                if missing_boolean: 
                    encodings = encodings[views_available] #drop encodings from views that are not available
                for i in range(n_views_available):
                    views_emb[i] += encodings[i]    
            views_stacked = self.stacker_function(views_emb)   
            n_batch, n_views, n_dims = views_stacked.shape       
            
        if self.mode in STACK_FUNC_NAMES:
            joint_emb_views = self.concater_function(views_emb)

        elif self.mode.split("_")[0] in POOL_FUNC_NAMES + ["rnn", "lstm", "gru", "transformer", "trans"]:
            if self.mode.split("_")[0] in ["rnn", "lstm", "gru"]:
                if missing_boolean and n_views_available != n_views: 
                    views_stacked = views_stacked[:, views_available, :]
                    permu_opts = self.permu_opts_viewsfunc(n_views_available)
                else:
                    permu_opts = self.permu_opts_viewsfunc(n_views) 

                if self.permute_rnn: 
                    indx_rnds = [np.random.randint(len(permu_opts))] if self.random_permute else np.arange(len(permu_opts))
                else:
                    indx_rnds = [0]
                views_stacked = torch.concat([views_stacked[:,permu_opts[indx_rnd], :] for indx_rnd in indx_rnds], axis=0)
                
            if self.mode.split("_")[0] in ["transformer", "trans"]: 
                if missing_boolean and n_views_available != n_views:
                    views_stacked = views_stacked[:, views_available, :]

            joint_emb_views = self.pooler_function(views_stacked)["rep"]
                        
            if self.mode.split("_")[0] in ["rnn", "lstm", "gru"] and len(indx_rnds) > 1: #pool from permutations
                joint_emb_views = torch.mean(joint_emb_views.reshape(len(indx_rnds), n_batch, n_dims), axis =0)
                
        if self.adaptive:
            att_views = self.attention_function(joint_emb_views)
            att_views = torch.reshape(att_views, (att_views.shape[0], n_views, n_dims)) if self.features else att_views[:,:, None]
            
            if missing_boolean:                  
                views_available = (views_available[None, :, None]).repeat(att_views.shape[0], 1, att_views.shape[-1])
                att_views[~views_available] = -torch.inf
            att_views = nn.functional.softmax(att_views, dim=1)

            joint_emb_views = torch.sum(views_stacked*att_views, dim=1)

        if self.mode.split("_")[0] in ["sampling"]:             
            probabilities = torch.ones(len(n_views), dtype=torch.float, device=views_stacked.device)
            if missing_boolean and n_views_available != n_views: 
                probabilities[~views_available] = 0
            probabilities = probabilities / probabilities.sum()
            
            probabilities = probabilities.repeat(n_batch, 1)
            if self.features:
                selection = torch.multinomial(probabilities, n_dims, replacement=True)[:,None,:]  #n_samples, 1, n_dims -- a selection for each sample-dimension config
            else:
                selection = torch.multinomial(probabilities, 1, replacement=True)[:,:,None] #n_samples, 1, 1 -- single selection for each sample

            joint_emb_views = views_stacked.gather(1, selection.expand(n_batch,n_views,n_dims))[:,0, :] #select the view features from the corresponding index
            
        dic_return = {"joint_rep": joint_emb_views}
        if self.adaptive:
            dic_return["att_views"] = att_views
        return dic_return

    def get_info_dims(self):
        return { "emb_dims":self.emb_dims, "joint_dim":self.joint_dim, "feature_pool": self.feature_pool}

    def get_joint_dim(self):
        return self.joint_dim
