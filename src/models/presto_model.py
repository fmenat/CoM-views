#from https://github.com/nasaharvest/presto
import presto
from presto import Presto
import torch

def match_inside(string, list_string):
    if len(list_string) == 0:
        return True
    for v in list_string:
        if "_" in v:
            if string in v.split("_"):
                return True
        if string == v:
            return True

class Presto_adapted(torch.nn.Module):
    def __init__(self, fine_tune=False, view_bands_order= [], latlons = None, num_outputs = 1):
        super(Presto_adapted, self).__init__()
        self.model = Presto.load_pretrained()
        self.model_encoder = self.model.encoder
        self.model_finetuning = self.model.construct_finetuning_model(num_outputs=num_outputs, regression=True)

        self.fine_tune = fine_tune
        self.view_bands_order = view_bands_order
        self.global_latlons = latlons if latlons is not None else [0,0]

    def forward(self, x, inference_views=[]):
        #x is an already concatenated array in this pipeline
        original_device = x.device
        global_month = 0 
        global_latlons = torch.Tensor(self.global_latlons).repeat(x.shape[0],1).float().to(original_device)

        #inefficient
        x_s = []
        mask_s = []
        dw_s = []
        for s in x:
            aux_x, mask, dw = presto.construct_single_presto_input(
                s1 = s[:,self.view_bands_order.get("s1")].to("cpu") if ("s1" in self.view_bands_order and match_inside("S1", inference_views)) else None,
                s1_bands= self.view_bands_order.get("s1_bands"),
                s2 = s[:,self.view_bands_order.get("s2")].to("cpu")  if ("s2" in self.view_bands_order and match_inside("S2", inference_views)) else None,
                s2_bands = self.view_bands_order.get("s2_bands"),
                era5 = s[:,self.view_bands_order.get("era5")].to("cpu")  if ("era5" in self.view_bands_order and match_inside("weather", inference_views)) else None,
                era5_bands = self.view_bands_order.get("era5_bands"),
                srtm = s[:,self.view_bands_order.get("srtm")].to("cpu")  if ("srtm" in self.view_bands_order and match_inside("DEM", inference_views)) else None,
                srtm_bands = self.view_bands_order.get("srtm_bands"),
                dynamic_world = None,
                normalize = True)
            x_s.append(aux_x)
            mask_s.append(mask)
            dw_s.append(dw)
        x = torch.stack(x_s, dim=0).float().to(original_device)
        mask = torch.stack(mask_s, dim=0).bool().to(original_device)
        dw = torch.stack(dw_s, dim=0).long().to(original_device)

        if self.fine_tune:
            return self.model_finetuning(x, dynamic_world=dw, mask=mask, latlons=global_latlons, month=global_month)
        else:
            return self.model_encoder(x, dynamic_world=dw, mask=mask, latlons=global_latlons, month=global_month, eval_task=True)
