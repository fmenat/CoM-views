import copy
import numpy as np
import torch
import pytorch_lightning as pl

from src.datasets.utils import _to_loader
from src.models.fusion_strategy import FeatureFusion
from src.models.fusion_module import FusionModuleMissing
from src.models.nn_models import create_model
from src.models.utils import get_dic_emb_dims
from src.training.pl_utils import prepare_callback

def assign_multifusion_name(training = {}, method = {}, forward_views= [], perc:float=1,  more_info_str = ""):
    #just to have a unique name for the method storage
    method_name = f"Feat_{method['agg_args']['mode']}"
    
    if "adaptive" in method["agg_args"]:
        if method["agg_args"]["adaptive"]:
            method_name += "_GF"
    if "features" in method["agg_args"]:
        if method["agg_args"]["features"]:
            method_name += "f"

    if "multi" in training["loss_args"]:
        if training["loss_args"]["multi"]:
            method_name += "_MuLoss"

    if training.get("missing_as_aug"):
        if training["missing_method"].get("missing_random"):
            method_name += "-SD"
        else:
            method_name += "-MAUG"
    if training.get("missing_method"):
        method_name += f"-{training['missing_method']['name']}" + (f"_{training['missing_method']['where']}" if training['missing_method'].get("where") else "")
    
    if len(forward_views) != 0:
        method_name += "-Forw_" + "_".join(forward_views)
        if perc != 1 and perc != 0:
            method_name += f"_{perc*100:.0f}"

    return method_name + more_info_str


def MultiFusion_train(train_data: dict, val_data = None, 
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="", 
                     training = {}, method = {}, architecture={}, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    method_name = assign_multifusion_name(training, method) if method_name == "" else method_name
    folder_c = output_dir_folder+"/run-saves"

    if "weight" in loss_args:
        n_labels = np.max(train_data["target"]) +1
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    else:
        n_labels = 1
    args_model = {"loss_args": loss_args, **training.get("additional_args", {})}

    #MODEL DEFINITION -- Encoder-Part
    views_encoder  = {}
    for i, view_n in enumerate(train_data["view_names"]):
        views_encoder[view_n] = create_model(train_data["views"][i].shape[-1], emb_dim, **architecture["encoders"][view_n])
    #MODEL DEFINITION -- Fusion-Part
    method["agg_args"]["emb_dims"] = get_dic_emb_dims(views_encoder)
    fusion_module = FusionModuleMissing(**method["agg_args"])
    input_dim_task_mapp = fusion_module.get_info_dims()["joint_dim"]
    #MODEL DEFINITION -- Predictive-Part
    predictive_model = create_model(input_dim_task_mapp, n_labels, **architecture["predictive_model"], encoder=False)  #default is mlp
    
    #MODEL DEFINITION -- Full Model
    model = FeatureFusion(views_encoder, fusion_module, predictive_model,view_names=list(views_encoder.keys()), **args_model)
    print("Initial parameters of model:", model.hparams_initial)
    if "missing_as_aug" in training:
        model.set_missing_info(aug_status=training["missing_as_aug"], **training.get("missing_method", {}))
            
    #DATA DEFINITION 
    if type(val_data) != type(None):
        val_dataloader = _to_loader(val_data, batch_size=batch_size, train=False)
        monitor_name = "val_objective"
    else:
        val_dataloader = None
        monitor_name = "train_objective"
    train_dataloader = _to_loader(train_data, batch_size=batch_size)
    
    #TRAINING 
    extra_objects = prepare_callback(data_name, method_name, run_id, fold_id, folder_c, model.hparams_initial, monitor_name, **early_stop_args)
    trainer = pl.Trainer(max_epochs=max_epochs, accelerator="gpu", devices = 1, callbacks=extra_objects["callbacks"])
    trainer.fit(model, train_dataloader, val_dataloaders=val_dataloader)
    #RETRIEVE BEST MODEL
    checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
    model.load_state_dict(checkpoint['state_dict'])

    return model, trainer
