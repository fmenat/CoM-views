import numpy as np
import pandas as pd
import argparse
import os
from tqdm import tqdm

from utils import storage_set

INPUT_FEATURES = {
    "conditions": ["DEWP","TEMP", "HUMI"],
    "dynamics": ["PRES", "cbwd", "Iws","season"],
    "precipitation": ["precipitation", "Iprec"],
    "pretarget": ["PM_US Post"]
    }
encode_ = {"NW": 315, "NE": 45, "cv": -1, "SE":135, "SW":225, np.nan: -360}

def create_ts(df, lag, col_target, cols_input, additionals, start_indx=0):
    indx_ = []
    target_data = []

    array_add = {v: [] for v in additionals}
    view_data = dict(**array_add, **{col: [] for col in cols_input.keys()})
    for t_plus1 in range(lag, len(df)):
        target = df[col_target].iloc[t_plus1]
        if not pd.isna(target):
            target_data.append([np.float32(target)])
        else:
            continue

        indx_.append(df["No"].iloc[t_plus1]+start_indx)    
        for v in additionals:
        	view_data[v].append([df[v].iloc[t_plus1]])
        
        for key, value in cols_input.items():
            ts_data = df[value].iloc[t_plus1-lag: t_plus1].values.astype(np.float32)
            view_data[key].append(ts_data)
        
    for key in view_data.keys():
        view_data[key] = np.asarray(view_data[key])
    
    return indx_, view_data, target_data

if __name__ == "__main__":
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument(
	    "--data_dir",
	    "-d",
	    required=True,
	    type=str,
	    help="path of the data directory",
	)
	arg_parser.add_argument(
	    "--out_dir",
	    "-o",
	    required=True,
	    type=str,
	    help="path of the output directory to store the data",
	)
	arg_parser.add_argument(
	    "--days",
	    "-D",
	    required=True,
	    type=int,
	    help="the number of days to be used in the prediction",
	    default= 3,
	)
	args = arg_parser.parse_args()


	print(f"CREATING AND SAVING DATA")
	indx_data, views_data, target_data = [], {}, []
	current_idx = 0
	for f in tqdm(os.listdir(args.data_dir)):
		if ".csv" in f:
			dat = pd.read_csv(args.data_dir+f)
			dat["cbwd"] = dat["cbwd"].apply(lambda x: encode_[x]) #cbwd to degrees

			indx_, views_, target_ = create_ts(dat, lag=24*args.days, start_indx=current_idx,
				col_target="PM_US Post", cols_input=INPUT_FEATURES, additionals=["year"])

			indx_data.extend(indx_)
			target_data.extend(target_)
			for col in views_.keys():
				if col in views_data:
					views_data[col].append(views_[col])
				else:
					views_data[col] = [views_[col]]
			current_idx+= len(dat)+1

	for col in views_data.keys():
	 	views_data[col] = np.concatenate(views_data[col], axis=0)

	storage_set([indx_data, views_data, np.array(target_data)], args.out_dir, 
		name=f"PM25_D{args.days}_train", mode="", full_view_flag=False, target_names="pm25")