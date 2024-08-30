from pathlib import Path
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint

def prepare_callback(data_name, method_name, run_id, fold_id, folder_c, tags_ml, monitor_name, **early_stop_args):
    save_dir_chkpt = f'{folder_c}/checkpoint_logs/'
    exp_folder_name = f'{data_name}/{method_name}'

    for v in Path(f'{save_dir_chkpt}/{exp_folder_name}/').glob(f'r={run_id:02d}_{fold_id:02d}*'):
        v.unlink()
    early_stop_callback = EarlyStopping(monitor=monitor_name, **early_stop_args)
    checkpoint_callback = ModelCheckpoint(monitor=monitor_name, mode=early_stop_args["mode"], every_n_epochs=1, save_top_k=1,
        dirpath=f'{save_dir_chkpt}/{exp_folder_name}/', filename=f'r={run_id:02d}_{fold_id:02d}-'+'{epoch}-{step}-{val_objective:.2f}')
    tags_ml = dict(tags_ml,**{"data_name":data_name,"method_name":method_name})
    return {"callbacks": [early_stop_callback,checkpoint_callback] }