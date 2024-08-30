import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm

import cropharvest
from cropharvest.datasets import CropHarvestLabels, CropHarvest, Task
from cropharvest.bands import BANDS

from utils import codify_labels, storage_set

FEATURES_CROPHARVEST = {
		"S2": ['B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'],
		"S1": ["VV", "VH"],
		"weather": ['temperature_2m',  'total_precipitation'],
		"DEM": [ 'elevation','slope'],
		"S2VI": ["NDVI"],
		"S1Ind": ["VH-VV"],
}

def extract_data(metadata_cropclass, data_structure):
	"""
		Function to extract the training and testing data for each set
	"""
	indexs , X_full_train, Y_full_train = [], [], []
	for i, value in enumerate(metadata_cropclass.index):
		new_detailed_label = metadata_cropclass.loc[value]["classification_label"]
		if not pd.isna(new_detailed_label):
			x_i, y_i = data_structure[i]
			X_full_train.append(x_i)
			Y_full_train.append(new_detailed_label)
			indexs.append(value)

	X_full_train = np.asarray(X_full_train)
	Y_full_train = np.asarray(Y_full_train)

	return indexs, X_full_train, Y_full_train

def extract_labels(DATA_DIR, indx_data):
	metadata_structure = CropHarvestLabels(DATA_DIR)

	metadata_structure._labels["year"] = metadata_structure._labels["collection_date"].apply(lambda x: x.split("-")[0])
	df_metadata = metadata_structure._labels
	final_indx = []
	for v in df_metadata.to_dict(orient="records"):
		dataset = v["dataset"]
		indx_list = v["index"] #read index directly from data
		final_indx.append(f"{indx_list}_{dataset}")
	df_metadata["indx"] = final_indx

	df_metadata = df_metadata.set_index("indx").loc[indx_data] #filter based on data available in xarray
	df_metadata = df_metadata[~df_metadata["is_test"]] #filter test data
	return df_metadata

def create_full_views(X_data):
	"""
		This function assumes that the features/channels of input data X comes in the same order as FEATURES_CROPHARVEST
	"""
	view_data = {}
	for feat in tqdm(FEATURES_CROPHARVEST):
		if "S1Ind" not in feat:
			idxs_feat = [np.where(np.asarray(BANDS) == feat_i)[0][0] for feat_i in FEATURES_CROPHARVEST[feat]]
			view_data[feat] = X_data[:,:,idxs_feat]
		else:
			view_data[feat] = view_data["S1"][:,:,1] - view_data["S1"][:,:,0]
		
		view_data[feat] = view_data[feat].astype(np.float32) #everything as float since there are some interpolation
	return view_data

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
	    "--fusion",
	    "-f",
	    required=False,
	    type=str,
	    help="type of fusion to be used, options are [fusion, any other]",
		default="any"
	)
	arg_parser.add_argument(
	    "--crops",
	    "-c",
	    required=False,
	    type=str,
	    help="whether use multi crop or not, options [multi, binary]",
	    default="binary"
	)
	args = arg_parser.parse_args()

	print("CREATING AND SAVING DATA")
	data = CropHarvest(args.data_dir, download=True)
	data.task.normalize = False
	indx_data = [str(f).split("/")[-1].split('.h5')[0] for f in data.filepaths]
	df_metadata = extract_labels(args.data_dir, indx_data)
		
	if args.crops == "binary":
		X_full_train, Y_full_train = data.as_array(flatten_x=False)

		target_data = Y_full_train.astype(np.int8)
		views_data = create_full_views(X_full_train)
		print(f"Dataset with {len(views_data)} views, and and shape",{v: views_data[v].shape for v in views_data})
		
		views_data["year"] = df_metadata["year"].values.astype(np.int16).reshape(-1,1)
		views_data["coords"] = df_metadata[["lon","lat"]].values.astype(np.float32)
		
		storage_set([indx_data, views_data, target_data], args.out_dir, 
			name="cropharvest_binary", mode=args.fusion, target_names="target crop", full_view_flag=True)

	elif args.crops == "multi":
		indx_data, X_full_train, Y_full_train = extract_data(df_metadata, data)

		views_data = create_full_views(X_full_train)
		LABELS = df_metadata["classification_label"].dropna().unique()
		target_data = codify_labels(Y_full_train, LABELS).astype(np.int8)
		print(f"Dataset with {len(views_data)} views, and and shape",{v: views_data[v].shape for v in views_data})

		views_data["coords"] = df_metadata[["lon","lat"]].loc[indx_data].values.astype(np.float32)
		views_data["year"] = df_metadata["year"].loc[indx_data].values.astype(np.uint16)

		storage_set([indx_data, views_data, target_data], args.out_dir, 
			name="cropharvest_multi_train", target_names=LABELS, mode=args.fusion, full_view_flag=True)
	else:
		raise Exception("Unrecognized type of crop-task")
