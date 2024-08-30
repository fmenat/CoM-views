
import torch

class TemporalDropout(torch.nn.Module):
    def __init__(self, p, correction: bool = False, replace_value=0, pos_start =0, pos_end = -1,  **kwargs):
        """
            Temporal Dropout: Dropout applied on specific time-steps. It mask a complete time-step vector.
            Input shape: (num_samples, num_timesteps, num_dims)
            Output shape: (num_samples, num_timesteps, num_dims) << where some timesteps will be all filled with *replace_value* 

            Reference: https://www.sciencedirect.com/science/article/pii/S0924271622000855
            Parameters
            ----------
            p: float
                the dropout ratio. A value between 0 and 1 to represent the ammount of time-steps that will be masked out
            correction: boolean
                whether to include the correction from Dropout paper (https://dl.acm.org/doi/abs/10.5555/2627435.2670313). Not necesarry for time-dropout
            replace_value: float
                a value that will replace the actual data based on the dropout.
                Ideally it should be the value used to mask missing information.
            pos_start: int
                from what index feature the temporal dropout will be performed on each time.
                By default all features, pos_start=0
            pos_end: int
                until what index feature the temporal dropout will be performed on each time.
                By default all features, pos_end=-1
            
        """
        super().__init__()
        self.tempdrop = torch.nn.Dropout1d(p=p)
        self.dropout_value = p
        self.correction = correction
        self.replace_value = replace_value
        self.pos_start = pos_start
        self.pos_end = pos_end

    def forward(self, x):
        if self.training:
            #select which features to mask out
            if self.pos_start != 0 or self.pos_end != -1:
                x_forward =  x[:,:,self.pos_start:self.pos_end] 
            else:
                x_forward = x

            #mask features on all times
            x_forward = self.tempdrop(x_forward)

            #correct/or not, and replace 0 with value
            if not self.correction:
                x_forward = x_forward*(1- self.dropout_value)
                if self.replace_value != 0:
                    x_forward[x_forward==0] = self.replace_value
            else:
                if self.replace_value != 0:
                    x_forward[x_forward==0] = self.replace_value/(1-self.dropout_value)

            if self.pos_start != 0 or self.pos_end != -1:
                x[:,:,self.pos_start:self.pos_end]  = x_forward
            else:
                x = x_forward
        return x
    

if __name__ == "__main__": #test
    data_x = torch.FloatTensor(5,3,5).uniform_(0,1)
    print("before dropout", data_x)

    model_ = torch.nn.Sequential(*[TemporalDropout(0.4), TemporalDropout(0.5)])
    print("after temporal dropout with correction (scaling)", model_(data_x))
    
    model_ = torch.nn.Sequential(*[TemporalDropout(0.4,correction=False), TemporalDropout(0.5,correction=False)])
    print("after temporal dropout without correction", model_(data_x))

    model_.eval()
    print("after temporal dropout in testing time (Should be the same)", model_(data_x))

    model_ = torch.nn.Sequential(*[TemporalDropout(0.4,correction=False, replace_value=65000)])
    print("after temporal dropout by replacing with 65000", model_(data_x))

    model_ = torch.nn.Sequential(*[TemporalDropout(0.4,correction=False, pos_start=1, pos_end=2)])
    print("temporal dropout on index 1", model_(data_x))