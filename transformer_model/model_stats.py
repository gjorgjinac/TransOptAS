from .tsai_custom import *
from tsai.all import *
computer_setup()

from torch.nn import CrossEntropyLoss, AvgPool2d,AdaptiveAvgPool2d

   
class OptTransStats(Module):
    def __init__(self, c_in:int, c_out:int, seq_len:int, iteration_count:int=None,
                 n_layers:int=3, d_model:int=128, n_heads:int=16, d_k:Optional[int]=None, d_v:Optional[int]=None,  
                 d_ff:int=256, dropout:float=0.1, act:str="gelu", fc_dropout:float=0.1,fc_size=50, use_positional_encoding=False,
                 y_range:Optional[tuple]=(0,1), verbose:bool=False, aggregations=None, do_regression=False, **kwargs):
        r"""TST (Time Series Transformer) is a Transformer that takes continuous time series as inputs.
        As mentioned in the paper, the input must be standardized by_var based on the entire training set.
        Args:
            c_in: the number of features (aka variables, dimensions, channels) in the time series dataset.
            c_out: the number of target classes.
            seq_len: number of time steps in the time series.
            max_seq_len: useful to control the temporal resolution in long time series to avoid memory issues.
            d_model: total dimension of the model (number of features created by the model)
            n_heads:  parallel attention heads.
            d_k: size of the learned linear projection of queries and keys in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_v: size of the learned linear projection of values in the MHA. Usual values: 16-512. Default: None -> (d_model/n_heads) = 32.
            d_ff: the dimension of the feedforward network model.
            dropout: amount of residual dropout applied in the encoder.
            act: the activation function of intermediate layer, relu or gelu.
            n_layers: the number of sub-encoder-layers in the encoder.
            fc_dropout: dropout applied to the final fully connected layer.
            y_range: range of possible y values (used in regression tasks).
            kwargs: nn.Conv1d kwargs. If not {}, a nn.Conv1d with those kwargs will be applied to original time series.
        Input shape:
            bs (batch size) x nvars (aka features, variables, dimensions, channels) x seq_len (aka time steps)
        """
        self.c_out, self.seq_len, self.do_regression = c_out, seq_len, do_regression
        
        # Input encoding
        self.new_q_len = False
        self.W_P = nn.Linear(c_in, d_model) # Eq 1: projection of feature vectors onto a d-dim vector space
        
        self.use_positional_encoding=False

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

        # Encoder
        self.encoder = TSTEncoder(seq_len, d_model, n_heads, d_k=d_k, d_v=d_v, d_ff=d_ff, dropout=dropout, activation=act, n_layers=n_layers)
        #self.pooling = torch.nn.AdaptiveAvgPool1d(1)
        
        # Head
        self.head_nf = d_model 
        if not do_regression:
            self.head=torch.nn.Sequential(
                torch.nn.Linear(self.head_nf,fc_size, bias=True),
                torch.nn.ReLU(),
                Flatten(),
                nn.Dropout(fc_dropout),
                torch.nn.Linear(fc_size, c_out, bias=True),
                SigmoidRange(*y_range)
            )
        else:
            self.head=torch.nn.Sequential(
                torch.nn.Linear(self.head_nf,fc_size, bias=True),
                torch.nn.ReLU(),
                Flatten(),
                nn.Dropout(fc_dropout),
                torch.nn.Linear(fc_size, c_out, bias=True),
                SigmoidRange(*y_range)
               )




    def get_embeddings(self, x:Tensor):

        # Input encoding
        if self.new_q_len: 
            u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else:
            u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        # Positional encoding
        if self.use_positional_encoding:
            u = self.dropout(u + self.W_pos)

        # Encoder
        z = self.encoder(u)                                             # z: [bs x q_len x d_model]
        #z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]
        z=z.mean(dim=1).squeeze()
        return z

    def forward(self, x:Tensor) -> Tensor:  # x: [bs x nvars x q_len]
        '''classification_token=torch.nn.Parameter(torch.rand(x.shape[0],x.shape[1], 1, device='cuda'))

        x=torch.concat([classification_token ,x],dim=-1)'''

        if self.new_q_len: 
            u = self.W_P(x).transpose(2,1) # Eq 2        # u: [bs x d_model x q_len] transposed to [bs x q_len x d_model]
        else: 
            u = self.W_P(x.transpose(2,1)) # Eq 1                     # u: [bs x q_len x nvars] converted to [bs x q_len x d_model]

        if self.use_positional_encoding:
            u = self.dropout(u + self.W_pos)
        #print(x.shape)
        # Encoder
        z = self.encoder(u)
        #z=z[:,0,:]

        #z = z.transpose(2,1).contiguous()                               # z: [bs x d_model x q_len]
        #z=self.pooling(z.transpose(1, 2)).squeeze()

        z=z.mean(dim=1).squeeze()
        #z = self.get_stats(z)
        # Classification/ Regression head
        z=self.head(z)
        return z