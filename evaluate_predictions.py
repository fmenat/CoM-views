import yaml
import argparse
import gc
import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rc('font', **{"size":14})

from src.datasets.views_structure import load_structure
from src.evaluate.utils import load_data_per_fold, save_results, gt_mask
from src.evaluate.vis import plot_prob_dist_bin, plot_conf_matrix, plot_dist_bin,plot_true_vs_pred
from src.metrics.metrics import ClassificationMetrics, SoftClassificationMetrics, RegressionMetrics

def classification_metric(
                preds_p_run,
                indexs_p_run,
                data_ground_truth,
                ind_save,
                set_display = False,
                include_metrics = [],
                dir_folder = "",
                task_type="classification",
                create_plots=False
                ):
    R = len(preds_p_run)

    df_runs = []
    df_runs_diss = []
    df_per_run_fold = []
    y_true_concatenated = []
    y_pred_cate_concatenated = [] #for classification
    for r in tqdm(range(R)):
        indexs_p_run_r = indexs_p_run[r]
        preds_p_run_r = preds_p_run[r]
        df_per_fold = []
        for f in tqdm(range(len(indexs_p_run_r))):
            y_true, y_pred = gt_mask(data_ground_truth, indexs_p_run_r[f]), preds_p_run_r[f]
            y_true = np.squeeze(y_true)
            y_pred = np.squeeze(y_pred)

            y_true_concatenated.append(y_true)

            if task_type == "classification":
                y_pred_cate = np.argmax(y_pred, axis = -1).astype(np.uint8)
                y_pred_cate_concatenated.append(y_pred_cate)
                
                d_me = ClassificationMetrics()
                dic_res = d_me(y_pred_cate, y_true)

                d_me_aux = SoftClassificationMetrics()
                dic_res.update(d_me_aux(y_pred, y_true))
                
                d_me = ClassificationMetrics(["F1 none", "R none", "P none", "ntrue", 'npred'])
                dic_des = d_me(y_pred_cate, y_true)
                df_des = pd.DataFrame(dic_des)
                df_des.index = ["label-"+str(i) for i in range(len(dic_des["N TRUE"]))]
                df_runs_diss.append(df_des)
            
                if "f1 bin" in include_metrics:
                    dic_res["F1 bin"] = dic_des["F1 NONE"][1]
                if "p bin" in include_metrics:
                    dic_res["P bin"] = dic_des["P NONE"][1]
            else:
                d_me = RegressionMetrics()
                dic_res = d_me(y_pred, y_true.astype(np.float32))
                
            df_res = pd.DataFrame(dic_res, index=["test"]).astype(np.float32)
            df_runs.append(df_res)

            df_per_fold.append(pd.DataFrame(dic_res, index=[f"fold-{f:02d}"]).astype(np.float32))

            if set_display:
                print(f"Run {r} being shown")
                print(df_res.round(4).to_markdown())
            del dic_res
            gc.collect()

        aux_ = pd.concat(df_per_fold).reset_index()
        aux_["run"] = [f"run-{r:02d}" for _ in range(len(indexs_p_run_r))]
        df_per_run_fold.append(aux_.set_index(["run","index"]))

    save_results(f"{dir_folder}/plots/{ind_save}/results_all", pd.concat(df_per_run_fold))
        
    df_concat = pd.concat(df_runs).groupby(level=0)
    df_mean = df_concat.mean()
    df_std = df_concat.std()

    save_results(f"{dir_folder}/plots/{ind_save}/preds_mean", df_mean)
    save_results(f"{dir_folder}/plots/{ind_save}/preds_std", df_std)
    print(f"################ Showing the {ind_save} ################")
    print(df_mean.round(4).to_markdown())
    print(df_std.round(4).to_markdown())

    if task_type == "classification":
        df_concat_diss = pd.concat(df_runs_diss).groupby(level=0)
        df_mean_diss = df_concat_diss.mean()
        df_std_diss = df_concat_diss.std()

        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_mean", df_mean_diss)
        save_results(f"{dir_folder}/plots/{ind_save}/preds_ind_std", df_std_diss)

    if create_plots:
        y_pred_concatenated = np.concatenate([np.concatenate(v) for v in preds_p_run],axis=0)
        y_true_concatenated = np.concatenate(y_true_concatenated,axis=0)
        
        if task_type == "classification":
            d_me = ClassificationMetrics(["confusion"])
            cf_matrix = d_me(np.concatenate(y_pred_cate_concatenated, axis=0), y_true_concatenated)["MATRIX"]        
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(6,4), squeeze=False)
            plot_conf_matrix(ax[0,0], cf_matrix)
            save_results(f"{dir_folder}/plots/{ind_save}/confusion_matrix_overall", plt)

            if len(d_me.n_samples) == 2:
                fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
                plot_prob_dist_bin(ax[0,0], y_pred_concatenated, y_true_concatenated)
                save_results(f"{dir_folder}/plots/{ind_save}/probabilities_overall", plt)

        else:
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,3), squeeze=False)
            plot_dist_bin(ax[0,0], y_pred_concatenated, y_true_concatenated)
            save_results(f"{dir_folder}/plots/{ind_save}/prediction_histogram", plt)
            
            fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(7,5), squeeze=False)
            plot_true_vs_pred(ax[0,0], y_pred_concatenated, y_true_concatenated)
            save_results(f"{dir_folder}/plots/{ind_save}/predictions_vs_groundtruth", plt)
        plt.close("all")
        plt.clf()

    return df_mean,df_std

def calculate_metrics(df_summary, df_std, data_te,data_name, method, task_type="classification", **args):
    preds_p_run_te, indexs_p_run_te = load_data_per_fold(data_name, method, **args)
    
    df_aux, df_aux2= classification_metric(
                        preds_p_run_te,
                        indexs_p_run_te,
                        data_te,
                        ind_save=f"{data_name}/{method}/",
                        task_type = task_type,
                        **args
                        )
    df_summary[method] = df_aux.loc["test"]
    df_std[method] = df_aux2.loc["test"]

def main_evaluation(config_file):
    input_dir_folder = config_file["input_dir_folder"]
    output_dir_folder = config_file["output_dir_folder"]
    data_name = config_file["data_name"]
    include_metrics = ["f1 bin", "p bin"]

    data_tr = load_structure(f"{input_dir_folder}/{data_name}.nc")

    if config_file.get("methods_to_plot"):
        methods_to_plot = config_file["methods_to_plot"]
    else:
        methods_to_plot = sorted(os.listdir(f"{output_dir_folder}/pred/{data_name}/"))

    df_summary_sup, df_summary_sup_s = pd.DataFrame(), pd.DataFrame()
    for method in methods_to_plot:
        print(f"Evaluating method {method}")
        calculate_metrics(df_summary_sup, df_summary_sup_s,
                        data_tr, 
                        data_name,
                        method,
                        include_metrics=include_metrics,
                        set_display=config_file.get("set_display"),
                        dir_folder=output_dir_folder,
                        task_type = config_file.get("task_type", "classification"),
                        create_plots=config_file.get("create_plots", False)
                        )
        gc.collect()

    print(">>>>>>>>>>>>>>>>> Mean across runs on test set")
    print((df_summary_sup.T).round(4).to_markdown())
    print(">>>>>>>>>>>>>>>>> Std across runs on test set")
    print((df_summary_sup_s.T).round(4).to_markdown())
    df_summary_sup.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_mean.csv")
    df_summary_sup_s.T.to_csv(f"{output_dir_folder}/plots/{data_name}/summary_std.csv")

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

    main_evaluation(config_file)
