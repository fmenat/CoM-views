
import numpy as np
import torch
import pytorch_lightning as pl
import copy

from src.datasets.utils import _to_loader
from src.models.nn_models import create_model
from src.models.fusion_strategy import InputFusion, SingleViewPool
from src.training.pl_utils import prepare_callback
from src.models.nn_models import create_awareness_vectors


def InputFusion_train(train_data: dict, val_data = None,
                data_name="", method_name="", run_id=0, fold_id=0, output_dir_folder="", 
                training={}, architecture= {}, **kwargs):
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]
    folder_c = output_dir_folder+"/run-saves"

    if "weight" in loss_args:
        n_labels = np.max(train_data["target"]) +1
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    else:
        n_labels = 1
    feats_dims = [v.shape[-1] for v in train_data["views"]]
    args_model = {"input_dim_to_stack": feats_dims, "loss_args": loss_args, **training.get("additional_args", {})}
    
    #Components Definition - Encoder and Predictive model
    encoder_model = create_model(np.sum(feats_dims), emb_dim, **architecture["encoders"])
    predictive_model = create_model(emb_dim, n_labels, **architecture["predictive_model"], encoder=False) 
    full_model = torch.nn.Sequential(encoder_model, predictive_model)
    
    #Full-model Definition
    model = InputFusion(predictive_model=full_model, view_names=train_data["view_names"], **args_model)
    print("Initial parameters of model:", model.hparams_initial)
    if "missing_as_aug" in training: 
        model.set_missing_info(aug_status=training["missing_as_aug"], **training.get("missing_method", {}))

    #DATA DEFITNION
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


def PoolEnsemble_train(train_data: dict, val_data = None,
                      data_name="", run_id=0, fold_id=0, output_dir_folder="", method_name="MuFu", 
                     training = {}, architecture={}, **kwargs):
    folder_c = output_dir_folder+"/run-saves"
    emb_dim = training["emb_dim"]
    max_epochs = training["max_epochs"]
    batch_size = training["batch_size"]
    early_stop_args = training["early_stop_args"]
    loss_args = training["loss_args"]

    if "weight" in loss_args:
        n_labels = np.max(train_data["target"]) +1
        loss_args["weight"] = torch.tensor(loss_args["weight"],dtype=torch.float)
    else:
        n_labels = 1
    if architecture["predictive_model"].get("awareness",{}).get("active", False):
        awareness_vectors = create_awareness_vectors(emb_dim, train_data["view_names"], init=architecture["predictive_model"]["awareness"].get("init", "zeros"))
        awareness_merge = architecture["predictive_model"]["awareness"].get("merge", "sum")
    else:
        awareness_vectors = {}
        awareness_merge =""
    args_model =  {"loss_args":loss_args, "view_names":train_data["view_names"], "awareness_vectors": awareness_vectors, "awareness_merge":awareness_merge, **training.get("additional_args", {})}
    
    #MODEL DEFINITION -- ENCODER and PREDICTIVE as a while
    pred_base = create_model(emb_dim if awareness_merge!="concat" else emb_dim*2, n_labels, **architecture["predictive_model"], encoder=False )  #default is mlp
    views_encoder  = {}
    view_prediction_heads = {}
    for i, view_n in enumerate(train_data["view_names"]):
        views_encoder[view_n] = create_model(train_data["views"][i].shape[-1], emb_dim, **architecture["encoders"][view_n])
        if architecture["predictive_model"].get("sharing"):
            view_prediction_heads[view_n] = pred_base
        else:
            view_prediction_heads[view_n] = copy.deepcopy(pred_base)
            view_prediction_heads[view_n].load_state_dict(pred_base.state_dict())   
    #MODEL DEFINITION -- Full Model
    model = SingleViewPool(views_encoder, view_prediction_heads, **args_model)
    print("Initial parameters of model:", model.hparams_initial)

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