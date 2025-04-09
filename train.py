import yaml
import argparse
import os
import sys
import time
import gc
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.utils import class_weight

from src.datasets.views_structure import DataViews, load_structure
from src.datasets.utils import _to_loader
from src.training.preprocess import preprocess_views
from src.training.pipeline import MultiFusion_train, assign_multifusion_name

def main_run(config_file, just_return_first_model=False):
    start_time = time.time()
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    view_names = config_file["view_names"]
    runs_seed = config_file["experiment"].get("runs_seed", [])
    if len(runs_seed) == 0:
        runs = config_file["experiment"].get("runs", 1)
        runs_seed = [np.random.randint(50000) for _ in range(runs)]
    kfolds = config_file["experiment"].get("kfolds", 2)
    preprocess_args = config_file["experiment"]["preprocess"]
    BS = config_file["training"]["batch_size"]
    
    data_views_all = load_structure(f"{input_dir_folder}/{data_name}", full_view_flag=config_file.get("full_view_flag", True))
    if "input_views" not in preprocess_args:
        preprocess_args["input_views"] = view_names
    preprocess_views(data_views_all, **preprocess_args)

    if "loss_args" not in config_file["training"]: 
        config_file["training"]["loss_args"] = {}
    if config_file.get("task_type", "").lower() == "classification":
        config_file["training"]["loss_args"]["name"] = "ce" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    elif config_file.get("task_type", "").lower() == "regression":
        config_file["training"]["loss_args"]["name"] = "mse" if "name" not in config_file["training"]["loss_args"] else config_file["training"]["loss_args"]["name"]
    method_name = assign_multifusion_name(config_file["training"],config_file["method"], more_info_str=config_file.get("additional_method_name", ""))

    metadata_r = {"epoch_runs":[], "prediction_time_full":[], "training_time":[], "best_score":[] }
    for r,r_seed in enumerate(runs_seed):
        np.random.seed(r_seed)
        indexs_ = data_views_all.get_all_identifiers() 
        if config_file["experiment"].get("group"): #stratified cross-validation
            name_group = config_file["experiment"].get("group")
            values_to_random_group = data_views_all.get_view_data(name_group)["views"]
            uniques_values_to_random_group = np.unique(values_to_random_group)
            np.random.shuffle(uniques_values_to_random_group)
            stratified_values_runs = np.array_split(uniques_values_to_random_group , kfolds)
            indexs_runs = []
            for stratified_values_r in stratified_values_runs:
                indxs_r = []
                for ind_i, value_i in zip(indexs_, values_to_random_group):
                    if value_i in stratified_values_r:
                        indxs_r.append(ind_i)
                indexs_runs.append(indxs_r)
        else: #regular random cross-validation
            np.random.shuffle(indexs_)
            indexs_runs = np.array_split(indexs_, kfolds)
        for k in range(kfolds):
            print(f"******************************** Executing model on run {r+1} and kfold {k+1}")
            
            data_views_all.set_test_mask(indexs_runs[k], reset=True)

            train_data = data_views_all.generate_full_view_data(train = True, views_first=True, view_names=view_names)
            val_data = data_views_all.generate_full_view_data(train = False, views_first=True, view_names=view_names)
            print(f"Training with {len(train_data['identifiers'])} samples and validating on {len(val_data['identifiers'])}")

            if config_file.get("task_type", "").lower() == "classification":
                train_data_target = train_data["target"].astype(int).flatten()
                config_file["training"]["loss_args"]["weight"]=class_weight.compute_class_weight(class_weight='balanced',classes= np.unique(train_data_target), y=train_data_target)
            
            #----------- Training -----------#
            start_aux = time.time()
            method, trainer = MultiFusion_train(train_data, val_data=val_data,run_id=r,fold_id=k,method_name=method_name, **config_file)
            if just_return_first_model:
                return method
            metadata_r["training_time"].append(time.time()-start_aux)
            metadata_r["epoch_runs"].append(trainer.callbacks[0].stopped_epoch)
            metadata_r["best_score"].append(trainer.callbacks[0].best_score.cpu())
            print("Training done")
            
            #---------- Store predictions of the model -----------#
            pred_time_Start = time.time()
            outputs_te = method.transform(_to_loader(val_data, batch_size=BS, train=False), out_norm=(config_file.get("task_type", "").lower() != "regression"))            
            metadata_r["prediction_time_full"].append(time.time()-pred_time_Start)
            data_save_te = DataViews([outputs_te["prediction"]], identifiers=val_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
            data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{method_name}", ind_views=True, xarray=False)
                
            #----------- Evaluating missing views scenarios -----------#
            if config_file.get("args_forward") and config_file["args_forward"].get("list_testing_views"): 
                for (test_views, percentages) in config_file["args_forward"].get("list_testing_views"):
                    for perc_missing in percentages:
                        print("Inference with the following views ",test_views, " and percentage missing ",perc_missing)
                        if "missing_method" in config_file["args_forward"]:
                            args_forward = {"inference_views":test_views, **{k:v for k,v in config_file["args_forward"].items() if k!= "list_testing_views"}}
                        else:
                            method.set_missing_info(None, **config_file["training"].get("missing_method", {}))
                            args_forward = {"inference_views":test_views, "missing_method": method.missing_method}
                            
                        pred_time_Start = time.time()
                        outputs_te = method.transform(_to_loader(val_data, batch_size=config_file['args_forward'].get("batch_size", BS), train=False), out_norm=(config_file.get("task_type", "").lower() != "regression"), args_forward=args_forward, perc_forward=perc_missing)
                        if f"prediction_time_{'_'.join(test_views)}_{perc_missing*100:.0f}" not in metadata_r:
                            metadata_r[f"prediction_time_{'_'.join(test_views)}_{perc_missing*100:.0f}"] = []
                        metadata_r[f"prediction_time_{'_'.join(test_views)}_{perc_missing*100:.0f}"].append(time.time()-pred_time_Start)

                        aux_name = assign_multifusion_name(config_file["training"],config_file["method"], forward_views=test_views, perc=perc_missing, 
                                                        more_info_str=config_file.get("additional_method_name", ""))
                        ## EXTRA -- PREDICTIONS ##
                        data_save_te = DataViews([outputs_te["prediction"]], identifiers=val_data["identifiers"], view_names=[f"out_run-{r:02d}_fold-{k:02d}"])
                        data_save_te.save(f"{output_dir_folder}/pred/{data_name}/{aux_name}", ind_views=True, xarray=False)
    
                        print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {aux_name} finished...")
            print(f"Fold {k+1}/{kfolds} of Run {r+1}/{len(runs_seed)} in {method_name} finished...")
    Path(f"{output_dir_folder}/metadata/{data_name}/{method_name}").mkdir(parents=True, exist_ok=True)
    pd.DataFrame(metadata_r).to_csv(f"{output_dir_folder}/metadata/{data_name}/{method_name}/metadata_runs.csv")
    print("Epochs for %s runs on average for %.2f epochs +- %.3f"%(method_name,np.mean(metadata_r["epoch_runs"]),np.std(metadata_r["epoch_runs"])))
    print(f"Finished whole execution of {len(runs_seed)} runs in {time.time()-start_time:.2f} secs")
    return metadata_r


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument(
        "--settings_file",
        "-s",
        action="store",
        dest="settings_file",
        required=True,
        type=str,
        help="path of the settings file",
    )
    args = arg_parser.parse_args()
    with open(args.settings_file) as fd:
        config_file = yaml.load(fd, Loader=yaml.SafeLoader)
    
    main_run(config_file)
