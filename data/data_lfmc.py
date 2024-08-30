import numpy as np
import pandas as pd
import argparse
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm
from utils import storage_set

INPUT_FEATURES = {
    "soil": [
    		"silt(t)","silt(t-1)", "silt(t-2)", "silt(t-3)",
             "sand(t)", "sand(t-1)", "sand(t-2)", "sand(t-3)",
             "clay(t)", "clay(t-1)", "clay(t-2)", "clay(t-3)"
            ], 
    "DEM": [
    		"slope(t)", "slope(t-1)", "slope(t-2)", "slope(t-3)",
            "elevation(t)", "elevation(t-1)", "elevation(t-2)", "elevation(t-3)",
           ],
    "LIDAR": ["canopy_height(t)", "canopy_height(t-1)", "canopy_height(t-2)", "canopy_height(t-3)"],
    "landcover": ["forest_cover(t)", "forest_cover(t-1)", "forest_cover(t-2)", "forest_cover(t-3)"],
    
    "S1": [
    		'vv(t-3)','vv(t-2)','vv(t-1)','vv(t)','vh(t-3)', 'vh(t-2)', 'vh(t-1)', 'vh(t)',
    		],
	"S1Ind": ['vh_vv(t-3)', 'vh_vv(t-2)', 'vh_vv(t-1)', 'vh_vv(t)'],
    "S2": [
    		'red(t-3)', 'red(t-2)', 'red(t-1)', 'red(t)', 'green(t-3)', 'green(t-2)', 'green(t-1)', 'green(t)','blue(t-3)', 'blue(t-2)', 'blue(t-1)', 'blue(t)', 
			  'swir(t-3)', 'swir(t-2)', 'swir(t-1)', 'swir(t)', 'nir(t-3)', 'nir(t-2)', 'nir(t-1)', 'nir(t)'
           ],
	"S2Ind": ['ndvi(t-3)', 'ndvi(t-2)', 'ndvi(t-1)', 'ndvi(t)', 'ndwi(t-3)', 'ndwi(t-2)', 'ndwi(t-1)', 'ndwi(t)','nirv(t-3)', 'nirv(t-2)', 'nirv(t-1)', 'nirv(t)'],
  
    "S1_S2": ['vv_red(t-3)', 'vv_red(t-2)', 'vv_red(t-1)', 'vv_red(t)',
              'vv_green(t-3)', 'vv_green(t-2)', 'vv_green(t-1)', 'vv_green(t)',
              'vv_blue(t-3)', 'vv_blue(t-2)', 'vv_blue(t-1)', 'vv_blue(t)',
              'vv_swir(t-3)', 'vv_swir(t-2)', 'vv_swir(t-1)', 'vv_swir(t)',
              'vv_nir(t-3)', 'vv_nir(t-2)', 'vv_nir(t-1)', 'vv_nir(t)',
              'vh_red(t-3)', 'vh_red(t-2)', 'vh_red(t-1)', 'vh_red(t)',
              'vh_green(t-3)', 'vh_green(t-2)', 'vh_green(t-1)', 'vh_green(t)',
              'vh_blue(t-3)', 'vh_blue(t-2)', 'vh_blue(t-1)', 'vh_blue(t)',
              'vh_swir(t-3)', 'vh_swir(t-2)', 'vh_swir(t-1)', 'vh_swir(t)',
              'vh_nir(t-3)', 'vh_nir(t-2)', 'vh_nir(t-1)', 'vh_nir(t)',
              
              'vv_ndvi(t-3)', 'vv_ndvi(t-2)', 'vv_ndvi(t-1)', 'vv_ndvi(t)',
              'vv_ndwi(t-3)', 'vv_ndwi(t-2)', 'vv_ndwi(t-1)', 'vv_ndwi(t)',
              'vv_nirv(t-3)', 'vv_nirv(t-2)', 'vv_nirv(t-1)', 'vv_nirv(t)',
              'vh_ndvi(t-3)', 'vh_ndvi(t-2)', 'vh_ndvi(t-1)', 'vh_ndvi(t)',
              'vh_ndwi(t-3)', 'vh_ndwi(t-2)', 'vh_ndwi(t-1)', 'vh_ndwi(t)',
              'vh_nirv(t-3)', 'vh_nirv(t-2)', 'vh_nirv(t-1)', 'vh_nirv(t)',
             ]
}

def extract_data(df_data):
	view_data = {}
	for feats in tqdm(INPUT_FEATURES):
		if "landcover" == feats or "LIDAR" == feats:
			enc = OneHotEncoder()
			data = enc.fit_transform(df_data[INPUT_FEATURES[feats]].values.flatten().reshape(-1,1)).toarray()
			print(f"Fitted with categories= {enc.categories_}")
			view_data[feats] = data.reshape(len(df_data), 4, -1).astype(np.uint8)
		else:
			data = df_data[INPUT_FEATURES[feats]].values
			view_data[feats] = data.reshape(len(df_data), -1, 4).transpose([0,2,1])

			if "soil" == feats:
				view_data[feats] = view_data[feats].astype(np.int16)
			else:
				view_data[feats] = view_data[feats].astype(np.float32) #s1 and s2 float only since index are inside

	indx_ = list(df_data.index)
	target_data = df_data["percent(t)"].values.astype(np.float32)

	print("features created are :" , list(view_data.keys()))
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
	    "--fusion",
	    "-f",
	    required=False,
	    type=str,
	    help="type of fusion to be used, options are [fusion, any other]",
		default="any"
	)
	args = arg_parser.parse_args()

	df_data = pd.read_csv(f"{args.data_dir}/training_features.csv", index_col=0)
	df_data= df_data[~df_data.index.duplicated(keep="last")] #remove ~30 duplicates

	print(f"CREATING AND SAVING DATA WITH FUSION MODE AS = {args.fusion}")
	indx_data, views_data, target_data = extract_data(df_data)

	views_data["coords"] = df_data[["longitude","latitude"]].values.astype(np.float32)
	views_data["year"] = pd.to_datetime(df_data["date"]).apply(lambda x: x.year).values.astype(np.uint16)
	
	storage_set([indx_data, views_data, target_data], args.out_dir,
		name="lfmc_train", mode = args.fusion,full_view_flag=True, target_names="lfmc")