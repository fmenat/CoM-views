import numpy as np
import sys
sys.path.insert(1, '../')

from src.datasets.views_structure import DataViews #to store data

STATIC_FEATS = ["soil", "dem", "lidar", "landcover"]

def codify_labels(array, labels):
    labels_2_idx = {v: i for i, v in enumerate(labels)} 
    return np.vectorize(lambda x: labels_2_idx[x])(array)
    
def storage_set(data, path_out_dir, name, mode = "input", target_names=[], full_view_flag=False):
	index, view_data, target = data
	print({"Index": len(index), "Target": len(target)}, {v: len(view_data[v]) for v in view_data})

	data_views = DataViews(full_view_flag=full_view_flag)
	for view_name in view_data:
		if mode.lower() == "input" and view_name.lower() in STATIC_FEATS: 
			data = view_data[view_name]
		else:
			if view_name.lower() in STATIC_FEATS:
				data = view_data[view_name][:,0,:]
			else:
				data = view_data[view_name]
			
		data_views.add_view(data, identifiers=index, name=view_name)
	data_views.add_target(target, identifiers=index, target_names=target_names)

	print(f"data stored in {path_out_dir}/{name}")
	add_info = "_input" if mode == "input" else ""
	data_views.save(f"{path_out_dir}/{name}{add_info}", xarray=True)