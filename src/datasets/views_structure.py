import copy, gc, sys, os , pickle
import numpy as np
import pandas as pd
import xarray as xray
from pathlib import Path
from typing import List, Union, Dict
from tqdm import tqdm

from sklearn.preprocessing import MultiLabelBinarizer

class DataViews(object):
    """a structure to handle the views in variable sence for the dataset,
    an instance could have variable number of views than other.
    calculate correlation between two arrays.
    
    Example: one item, one data example, could contain several views.
    n-views: number of views
    n-examples: number of examples
    
    Attributes
    ----------
        views_data : dictionary 
            with the data {key:name}: {view name:array of data in that view}
        views_data_ident2indx : dictionary of dictionaries
            with the data {view name: dictionary {index:identifier} }
        inverted_ident : dictionary 
            with {key:indx}: {identifier:list of views name (index) that contain that identifier}
        view_names : list of string
            a list with the view names
        views_cardinality : dictionary 
            with {key:name}: {view name: n-examples that contain that view}
        train_mask_identifiers : dictionary
            with boolean mask for each identifier if it is train or not
        identifiers_target: dictionary
            with the target corresponding ot each index example
        target_names : list of strings
            list with string of target names, indicated in the order
    """ 
    def __init__(self, views_to_add: Union[list, dict] = [], identifiers:List[int] = [], view_names: List[str] =[] , target: List[int] =[], full_view_flag=False):

        """initialization of attributes. You also could given the views to add in the init function to create the structure already, without using add_view method

        Parameters
        ----------
            views_to_add : list or dict of numpy array, torch,tensor or any
                the views to add
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the views_to_add
            view_names : list of string 
                the name of the views being added
            target: list of int
                the target values if available (e.g. supervised data)
        """

        ## if views to add given, it create the instance with views already saved
        self.views_data = {}
        self.views_data_ident2indx = {} 
        self.inverted_ident = {} 
        self.view_names = []
        self.views_cardinality = {} 
        self.train_mask_identifiers = {}
        self.identifiers_target = {}
        self.target_names = ["unsupervised"]
        self.full_view_flag = full_view_flag
        
        if len(views_to_add) != 0:
            if len(identifiers) == 0:
                identifiers = np.arange(len(views_to_add[0]))
            if len(view_names) == 0 and type(views_to_add) != dict:
                view_names = ["S"+str(v) for v in np.arange(len(views_to_add))]
            elif len(view_names) == 0 and type(views_to_add) == dict:
                view_names = list(views_to_add.keys())

            for v in range(len(views_to_add)):
                if type(views_to_add) == list or type(views_to_add) == np.ndarray:
                    self.add_view(views_to_add[v], identifiers, view_names[v])
                if type(views_to_add) == dict:
                    self.add_view(views_to_add[view_names[v]], identifiers, view_names[v])
            if len(target) != 0:
                self.add_target(target, identifiers)
        
    def add_target(self, target_to_add: Union[list,np.ndarray], identifiers: List[int], target_names: List[str] = [], update: bool =True):
        """add a target for the corresponding identifiers indicated, it also works by updating target

        Parameters
        ----------
            target_to_add : list, np.array or any structure that could be indexed as ARRAY[i]
                the target values to add 
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the target_to_add
            target_names : list of str 
                the target names if available (e.g. categorical targets)
            update: bool
                whether update the target when identifiers match an already saved value
        """
        for i, ident in enumerate(identifiers): 
            v = target_to_add[i]
            if type(v) == list: #or type(v) == np.ndarray:
                v = np.asarray(v)
            if type(v) == np.ndarray:
                if len(v.shape) == 1 and v.shape[0] == 1:
                    v = np.asarray([v.squeeze()])
                else:
                    v = v.squeeze()
            else:
                v = np.sarray(v)
            if ident not in self.identifiers_target:
                self.identifiers_target[ident] = v
            else:
                if (self.identifiers_target[ident] != target_to_add[i]) and update:
                    print("There are target that are being updated, if you dont want this behavior, set update=False")
                    self.identifiers_target[ident] = v
        if len(target_names) == 0:
            self.target_names = [f"T{i}" for i in range(len(self.identifiers_target[identifiers[-1]]))]
        else:
            self.target_names = target_names
        
    def get_target(self, identifiers: List[int] = [], matrix:bool=False) -> dict:
        """ get all the target associated to each identifiers

        Parameters
        ----------
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the view_to_add
        Returns
        -------
            dictionary of values
                "target": list of target for the corresponding identifiers
                "names": if the target is categorical and has available name, it will be returned also
        """

        if len(identifiers) == 0:
            target_dic = self.identifiers_target
        else:
            target_dic = {ident: self.identifiers_target[ident] for ident in identifiers}
        if matrix:
            target_dic = np.asarray(list(target_dic.values()))
        return {"target":target_dic, "target_names":self.target_names, "identifiers": identifiers}
            
    def flatten_views(self):
        """ flatten each view into a 2D-array (matrix)
        """
        for name in self.view_names:
            data = self.views_data[name]
            self.views_data[name] = data.reshape(data.shape[0], -1)

    def add_view(self, view_to_add, identifiers: List[int], name: str):
        """add a view array based on identifiers of list and name of the view. The identifiers is used to match the view with others views.

        Parameters
        ----------
            view_to_add : numpy array
                the array of the view to add (no restriction in dimensions or shape)
            identifiers : list of ints 
                each int will be the identifiers that correspond to each example in the view_to_add
            name : string 
                the name of the view being added
        """
        if name in self.view_names:
            print("The view is already saved, try update")
            return
        self.view_names.append(name)
        self.views_data[name] = np.asarray(view_to_add, dtype="float32") if type(view_to_add) != np.ndarray else view_to_add
        self.views_cardinality[name] = len(identifiers)

        #update inverted identifiers of items
        if self.full_view_flag:
            name = "full"

        if name not in self.views_data_ident2indx:
            self.views_data_ident2indx[name] = {}
            for indx, ident in enumerate(identifiers):
                if ident not in self.inverted_ident:
                    self.inverted_ident[ident] = [len(self.view_names)-1]
                    self.train_mask_identifiers[ident] = True
                else:
                    self.inverted_ident[ident].append(len(self.view_names)-1)
                self.views_data_ident2indx[name][ident] = indx
        
    def get_n_total(self):
        return len(self.identifiers_target)

    def __len__(self) -> int:
        return len(self.views_data)

    def __getitem__(self, i: int):
        """
        Parameters
        ----------
            i : int value that correspond to the example to get (with all the views available)

        Returns
        -------
            dictionary with three values
                data : numpy array of the example indicated on 'i' arg
                views : a list of strings with the views available for that example
                train? : a mask indicated if the example is used for train or not    
        """
        return self.get_item_ident(i)

    def get_item_ident(self, identifier: int):
        if identifier not in self.identifiers_target:
            raise Exception("identifier requested not available for the data in the DataViews class")
        if self.full_view_flag:
            views_available = self.view_names
            S_data = [self.views_data[view][self.views_data_ident2indx["full"][identifier]] for view in views_available]
        else:
            viewsname_based_ident = self.inverted_ident[identifier]
            if np.isnan(viewsname_based_ident).sum() > 0 : 
                viewsname_based_ident = np.asarray(viewsname_based_ident, dtype="int")[~np.isnan(viewsname_based_ident)].tolist()
            views_available = self.get_view_names(viewsname_based_ident)
            S_data = [self.views_data[view][self.views_data_ident2indx[view][identifier]] for view in views_available]
        return_info = {"views": S_data, 'view_names':views_available, 'train?': self.train_mask_identifiers[identifier], "identifier": identifier}
        
        return_info["target"] = self.identifiers_target[identifier]
        return return_info

    def filter_dict_by_views(self, views_data_dict: Dict[str, list], view_names_to_return: List[str]):
        return [views_data_dict[view_n] for view_n in view_names_to_return if view_n in views_data_dict ]
        

    def get_item_selected_views(self, identifier: int, view_names: List[str], return_info={}):
        """same as getitem method but masking on selected views
        Parameters
        ----------
            view_names : list of views to obtain the example 
            identifier : the identifier of the data to obtain
            
        Returns
        -------
            same as getitem method            
        """
        if len(return_info) == 0:
            return_info = self[identifier]
        view_n_data_dict = dict(zip(return_info["view_names"], return_info["views"]))
        return_info["views"] = self.filter_dict_by_views(view_n_data_dict, view_names)
        return_info["view_names"] = view_names
        return return_info

    def get_view_data(self, name: str):
        """get the numpy array of the view.
        The views inside the structure are not necesarrily in the same order, i.e. the examples contained are not related in the same axis. If you want to obtain same examples for each view run the generate_full_view_data method.

        Parameters
        ----------
            name : string with the name of the view

        Returns
        -------
            numpy array of the view indicated in 'name' param
            
        """
        return {"views":self.views_data[name], "identifiers": list(self.views_data_ident2indx["full"].keys()) if self.full_view_flag else list(self.views_data_ident2indx[name].keys()) , "view_names": [name]}

    def get_views_card(self, view_names: List[str]=[]):
        """get views cardinality or n-examples in each view.

        Parameters
        ----------
            view_names : list of string with the name of the views to calculate cardinality

        Returns
        -------
            a dictionary with the name of the indicated view in key and the cardinality in value
            
        """
        if len(view_names) == 0:
            return self.views_cardinality
        else:
            return {view_n: self.views_cardinality[view_n] for view_n in view_names}

    def get_all_identifiers(self) -> list:
        """get the identifiers of all views on the structure
     
        Returns
        -------
            list of identifiers
            
        """
        #set by train and test also
        return list(self.identifiers_target.keys())

    def get_view_names(self, indexs: List[int] = []) -> List[str]:
        """get the view names

        Parameters
        ----------
            indexs : if the index of which the view names will be returned
        
        Returns
        -------
            all the view names used in the structure
            
        """
        if len(indexs) == 0:
            return self.view_names
        else:
            return np.asarray(self.view_names)[indexs].tolist()

    def get_view_shapes(self, view_names: List[str]=[]):
        if len(view_names) == 0:
            view_names = self.view_names
        return {name: self.views_data[name].shape[1:] for name in view_names}
    def get_n_per_views(self, view_names: List[str]=[]):
        if len(view_names) == 0:
            view_names = self.view_names
        return {name: self.views_data[name].shape[0] for name in view_names}

    def generate_full_view_data(self, view_names: List[str]=[], stack: bool = False, views_first: bool=True, train: bool =True,  N:int =-1):
        """obtain a full-view dataset, i.e. get all the examples that contain all the views on the data structure.
        
        The return of the method is based on each example. return = n-samples x views x data dimension on each view

        Parameters
        ----------
            view_names : a list of strings for each of the views to generate the full view data. An example will be selected if all the views are available for that example. If not specified, all data views will be returned. Repeated views are ignored.
            stack: ??
            views_first: if the return array should contain the view as first dimension, as a list of views for the dataset
            train : if the examples to obtain should be training examples or not
            N: the total ammount of examples to sample, if -1, retrieve all.

        Returns
        -------
            dictionary with three values
                data : a list of numpy arrays with the examples, if view_first: len(data) = n-views (and n-views x n-examples x data dimension on each view), if not view_first: len(data) = n-examples (and n-examples x n-views x data dimension on each view)
                identifiers : a list of identifiers of the data returned
            
        """        
        if len(view_names) == 0:
            view_names = self.get_view_names()
        view_names = list(dict.fromkeys(view_names))
    
        additional_views = []
        for v in view_names:
            if "_" in v:
                view_names.extend(v.split("_"))
                additional_views.append(v)
        if len(additional_views) != 0:
            view_names = [v for v in view_names if v not in additional_views]
            
        print(f"You select {len(view_names)} views from the {len(self.view_names)} available, you could use get_view_names() to check which are available, the selected views are {view_names}, with additional views {additional_views}")

        if self.full_view_flag:
            view_less_data = "full"
        else:
            #search by the data with the least cardinality (view with less data)
            view_less_data =  sorted(self.get_views_card(view_names).items(), key=lambda x: x[1], reverse=True)[0][0]

        S_data, Idts_data, Y_data = [] , [], []
        if views_first:
            S_data = [[] for _ in range(len(view_names))]
            print("first dimension of list will be the views, instead of the standard of n-samples")
        for ident in tqdm(self.views_data_ident2indx[view_less_data].keys()): #just check the other views that has the query view
            if self.train_mask_identifiers[ident] == train:
                data_ = self[ident]
                V_ident, T_mask = data_["view_names"], data_["train?"]
                if all(view in V_ident for view in view_names): #check if data contain all the views
                    Idts_data.append(ident)
                    Y_data.append(data_["target"])
                    
                    if views_first:
                        data_ = self.get_item_selected_views(ident, view_names, return_info=data_)["views"]
                        for v in range(len(view_names)):
                            S_data[v].append(data_[v])
                    else:
                        S_data.append(self.get_item_selected_views(ident, view_names, return_info=data_)["views"])

        Idts_data = np.asarray(Idts_data)
        S_data = [np.asarray(S_data[v]) for v in range(len(view_names))] #transform each view into a fixed-size matrix
        Y_data = np.asarray(Y_data)
        if len(Y_data.shape) == 1:
            Y_data = np.expand_dims(Y_data, axis =-1)

        if N != -1:
            indx_sampled = np.random.choice( np.arange(len(Idts_data)), size=N, replace=False)
            Idts_data = Idts_data[indx_sampled]
            S_data = [S_data[v][indx_sampled] for v in range(len(view_names))]
            Y_data = Y_data[indx_sampled]
        
        if len(additional_views) != 0:
            for merge_views in additional_views:
                data_to_concat = []
                for v in merge_views.split("_"):
                    indx_v = np.where(np.asarray(view_names) == v)[0][0]
                    view_names.pop(indx_v)
                    data_to_concat.append(S_data.pop(indx_v))
                S_data.append( np.concatenate(data_to_concat, axis =-1))
                view_names.append(merge_views)

        if stack:
            print('stack set up, it is mandatory to be used only with same data dimension on all views')
            return {"views": np.concatenate(S_data, axis=-1), "identifiers": Idts_data, "target":Y_data, "view_names":view_names}
        else:
            
            return {"views": S_data, "identifiers": Idts_data, "target":Y_data, "view_names":view_names} 
        
    def set_test_mask(self, identifiers: List[int], reset = False):
        """set a binary mask to indicate the test examples

        Parameters
        ----------
            identifiers : list of identifiers that correspond to the test examples

        """
        if reset:
            self.train_mask_identifiers = dict.fromkeys(self.train_mask_identifiers, True )
        for v in identifiers:
            self.train_mask_identifiers[v] = False
            
    def view_train_test(self):
        N_train = np.sum(list(self.train_mask_identifiers.values()))
        N_test = len(self.train_mask_identifiers.values()) - N_train
        print(f'in total there is {N_train} train and {N_test} test, corresponding {N_train/(N_train+N_test)}/{N_test/(N_train+N_test)}')
        
    def apply_views(self, func):
        for view_name in self.view_names:
            if type(func) == dict:
                if view_name in func:
                    self.views_data[view_name] = func[view_name](self.views_data[view_name])  
            else:
                self.views_data[view_name] = func(self.views_data[view_name])  

    def _to_xray(self):
        data_vars = {}
        for view_n in self.get_view_names():
            data_vars[view_n] = xray.DataArray(data=self.views_data[view_n], 
                                  dims=["identifier"] +[f"{view_n}-D{str(i+1)}" for i in range (len(self.views_data[view_n].shape)-1)], 
                                 coords={"identifier": list(self.views_data_ident2indx["full"].keys()) if self.full_view_flag else list(self.views_data_ident2indx[view_n].keys()),
                                        #"dims": 
                                        }, )
        data_vars["train_mask"] = xray.DataArray(data=np.asarray(list(self.train_mask_identifiers.values()), dtype=bool),  #*1 to map to int
                                        dims=["identifier"], 
                                         coords={"identifier": list(self.train_mask_identifiers.keys()) })
        if len(self.identifiers_target) != 0:
            print(np.stack(list(self.identifiers_target.values()), axis=0).dtype)
            data_vars["target"] = xray.DataArray(data =np.stack(list(self.identifiers_target.values()), axis=0),
                dims=["identifier","dim_target"] , coords ={"identifier": list(self.identifiers_target.keys())} ) 

        if not self.full_view_flag:
            ohe_views = MultiLabelBinarizer().fit_transform(self.inverted_ident.values())
            data_vars["inverted_ident"] = xray.DataArray(data=ohe_views.astype(bool), dims=["identifier", "view_available"], 
                    coords={"identifier": list(self.inverted_ident.keys()) })
        
        return xray.Dataset(data_vars =  data_vars,
                        attrs = {"view_names": self.view_names, 
                                 "target_names": self.target_names,
                                },
                        )

    def save(self, name_path, xarray = True, ind_views = False):
        """save data in name_path

        Parameters
        ----------
            name_path : path to a file to save the model
            ind_views : if you want to save the individual views as csv files 
        """
        path_ = Path(name_path)
        name_path_, _, file_name_ = name_path.rpartition("/") 
        path_ = Path(name_path_)
        path_.mkdir(parents=True, exist_ok=True)
        if xarray and (not ind_views): 
            xarray_data = self._to_xray()
            path_ = path_ / (file_name_+".nc" if "nc" != file_name_.split(".")[-1] else file_name_)
            xarray_data.to_netcdf(path_, engine="h5netcdf") 
        elif (not xarray) and ind_views:  #only work with 2D array
            path_ = Path(name_path_ +"/"+ file_name_)
            path_.mkdir(parents=True, exist_ok=True)
            for view_name in self.get_view_names():
                view_data_aux = self.get_view_data(view_name)
                df_tosave = pd.DataFrame(view_data_aux["views"])
                df_tosave.index = view_data_aux["identifiers"]
                df_tosave.to_csv(f"{str(path_)}/{view_name}.csv", index=True)

    def load(self, name_path):
        """load data 

        Parameters
        ----------
            name_path : path to a file to save the model, without extension
        """
        return load_structure(name_path)


def load_structure(name_path: str, full_view_flag: bool= True):
    ext = name_path.split(".")[-1]
    if "nc" != ext:
        name_path= name_path+'.nc'
    data  = xray.open_dataset(name_path, engine="h5netcdf")
    return xray_to_dataviews(data, full_view_flag=full_view_flag)

def xray_to_dataviews(xray_data: xray.Dataset, full_view_flag: bool=True):
    all_possible_index = xray_data.coords["identifier"].values

    if full_view_flag:
        dataviews = DataViews(full_view_flag=full_view_flag)    
        dataviews.train_mask_identifiers = dict(zip(all_possible_index, xray_data["train_mask"].values.astype(bool)))
        dataviews.identifiers_target = dict(zip(all_possible_index, xray_data["target"].values))
        dataviews.views_data_ident2indx["full"] = dict(zip(all_possible_index,np.arange(len(all_possible_index))))

        dataviews.view_names = xray_data.attrs["view_names"]
        dataviews.target_names = xray_data.attrs["target_names"]
        for view_n in dataviews.view_names:
            #check nans cause of missingness
            data_variable = xray_data[view_n].dropna("identifier", how = "all") #variable array for each view

            dataviews.views_data[view_n] = data_variable.values
            dataviews.views_cardinality[view_n] = len(all_possible_index)
    else:
        dataviews = DataViews()
        for view_name in xray_data.attrs["view_names"]:	
            dataviews.add_view(xray_data[view_name].values, identifiers=all_possible_index, name=view_name)
        dataviews.add_target(xray_data["target"].values, identifiers=all_possible_index,target_names=xray_data.attrs["target_names"])
        dataviews.train_mask_identifiers = dict(zip(all_possible_index, xray_data["train_mask"].values.astype(bool)))
    return dataviews