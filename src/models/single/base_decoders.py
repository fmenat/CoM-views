from torch import nn
import abc

class Base_Decoder(abc.ABC, nn.Module):
    """
    Class to add methods for common modality specific methods
    """

    @abc.abstractmethod
    def get_output_size(self):
        pass

class Generic_Decoder(Base_Decoder):
    """
        it adds a prediction head (linear layer) with possible batch normalization to decoder layers.
    """
    def __init__(
        self,
        decoder: nn.Module,
        out_dims: int,
        use_norm: bool = False,
        **kwargs,
    ):
        super(Generic_Decoder, self).__init__()
        #self.return_all = False
        self.pre_decoder = decoder

        #build decoder head
        self.out_dims = out_dims
        self.use_norm = use_norm
        self.linear_layer = nn.Linear(self.pre_decoder.get_output_size(), self.out_dims)

    def forward(self, x, **kwargs):
        out_forward = self.pre_decoder(x) #should return a dictionary with output data {"rep": tensor}, or a single tensor
        if type(out_forward) != dict:
            out_forward = {"rep": out_forward}

        return_dic = {"rep": self.linear_layer(out_forward["rep"])}
        return return_dic["rep"] #single tensor output

    def get_output_size(self):
        return self.out_dims

    def update_first_layer(self, input_features):
        if hasattr(self.pre_decoder, "layers"):
            original_out_first = self.pre_decoder.layers[0][0].out_features
            self.pre_decoder.layers[0][0] = torch.nn.Linear(in_features=input_features, out_features=original_out_first)
        else:
            raise Exception(f"Trying to update first layer of decoder model but no *layers* were found in model {self.pre_decoder}")
