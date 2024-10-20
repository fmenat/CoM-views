import torch

from .base_encoders import Base_Encoder
from .transformer_utils import PositionalEncoding

class MLP(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_sizes: tuple = None,
        activation=torch.nn.ReLU, #LeakyReLU, GELU or nn.Tanh()
        dropout=0,
        batchnorm: bool=False,
        **kwargs,
    ):
        super(MLP, self).__init__()
        if layer_sizes is None:
            layer_sizes = (128,)
        layer_sizes = (feature_size,) + layer_sizes
        self.encoder_output = layer_sizes[-1]
        layers = []
        # other layers
        for l_id in range(len(layer_sizes) - 1):
            layers.append(
                torch.nn.Sequential(
                    torch.nn.Linear(layer_sizes[l_id], layer_sizes[l_id + 1]),
                    activation(),
                    torch.nn.BatchNorm1d(layer_sizes[l_id+1], affine=True) if batchnorm else torch.nn.Identity(),
                    torch.nn.Dropout(p=dropout) if dropout!=0 else torch.nn.Identity(),
                )
            )
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return {"rep": self.layers(x)}

    def get_output_size(self):
        return self.encoder_output

class RNNet(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_size: int = 128,
        dropout: float =0,
        num_layers: int = 1,
        bidirectional: bool = False,
        unit_type: str="gru",
        output_state: bool = False,
        **kwargs,
    ):
        super(RNNet, self).__init__()
        self.unit_type = unit_type.lower()
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.output_state = output_state

        if self.unit_type == "gru":
            rnn_type_class = torch.nn.GRU
        elif self.unit_type == "lstm":
            rnn_type_class = torch.nn.LSTM
        elif self.unit_type == "rnn":
            rnn_type_class = torch.nn.RNN
        else:
            raise Exception("unit_type not recognized, available options are [rnn, gru, lstm]")

        self.rnn = rnn_type_class(
                input_size=self.feature_size,
                hidden_size=self.layer_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional)
        
    def forward(self, x, **kwargs):
        rnn_out, states = self.rnn(x)
        states = states[0] if type(states) == tuple else states
        if self.bidirectional:
            rnn_out = rnn_out.view(-1, rnn_out.shape[1], 2, self.layer_size)
            rnn_out = (rnn_out[:,:, 0,: ] + rnn_out[:,:, 1,: ])/2

            if self.output_state:
                states = states.view(2, self.num_layers, -1, self.layer_size)
                states = (states[0] + states[1])/2

        if self.output_state: #get the last hidden state of the last layer
            out_ = states[-1]
        else: #get the last output 
            out_ = rnn_out[:, -1] 
        return {"rep": out_}

    def get_output_size(self):
        return self.layer_size


class TransformerNet(Base_Encoder):
    def __init__(
        self,
        feature_size: int,
        layer_size: int = 128,
        dropout: float = 0.0,
        num_layers: int = 1,
        num_heads: int = 1,
        len_max_seq: int = 24,
        fixed_pos_encoding: bool = True,
        **kwargs,
    ):
        super(TransformerNet, self).__init__()
        self.feature_size = feature_size
        self.layer_size = layer_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.len_max_seq = len_max_seq+ 1 #for class token
        self.num_heads = num_heads
        self.fixed_pos_encoding = fixed_pos_encoding

        self.cls_token = torch.nn.Parameter(torch.zeros(1, 1, self.feature_size))  
        self.pos_encoder = PositionalEncoding(
            d_model=self.feature_size, 
            dropout=self.dropout,
            max_len=self.len_max_seq)
        
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=self.feature_size,
            nhead=self.num_heads,
            dim_feedforward=self.layer_size,
            dropout=self.dropout,
            batch_first=True,
            **kwargs
        )
        self.tr_encoder = torch.nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=self.num_layers)

    def forward(self, x, src_key_padding_mask=None, **kwargs):
        # add cls token to input sequence
        x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), 1)

        # add time step for cls token in src_key_padding_mask
        if src_key_padding_mask is not None:
            src_key_padding_mask = torch.cat(
                (
                    torch.zeros_like(src_key_padding_mask[:, -1].reshape(-1, 1)).bool(),  #always false to attend cls token
                    src_key_padding_mask,  # for rest fo x
                ),
                dim=1)
        if self.fixed_pos_encoding: # add position encoding
            x = self.pos_encoder(x)
        
        # pass x through transformer encoder
        x = self.tr_encoder(x, src_key_padding_mask=src_key_padding_mask)
        
        return {"rep": x[:, 0, :]} # extract class token feature: 

    def get_output_size(self):
        return self.feature_size