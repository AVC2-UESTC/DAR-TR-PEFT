import math
import torch
import torch.nn as nn

from mmengine.model import kaiming_init, constant_init
from einops import rearrange

from ..utils import nchw_to_nlc, nlc_to_nchw
from ..utils import build_activation_layer
from ..utils import build_norm_layer
from ..utils import build_dropout




class Adapter(nn.Module):
    '''
    Args:
        in_channels: input channels $d$ e.g. C
        bottleneck (int): $\hat{d}$
        
        
        adapter_scalar (str): s. Could be learnable scalar i.e. 'learnable_scalar'
            Default: '1.0'
            
        act_cfg: activation function config. See src.Models.utils.
            Default: dict(
                act_type = 'ReLU', 
                inplace = False
            )
            
        adapter_layernorm_option: The position of norm layer.
            'in' for the norm before the down layer.
            'out' for the norm after the up layer.
            'none': no norm layer
            Default: 'in'
            
        norm_cfg: norm config.
            Default: dict(norm_type='LayerNorm', 
                                 requires_grad=True,
                                 eps=1e-5,
                                 affine=True)
            
        dropout_layer: dropout config
            Default: dict(drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
        
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    
    '''
    def __init__(self, 
                 in_channels, 
                 
                 # finetune cfg
                 bottleneck = None, 
                 
                 act_cfg = dict(act_type='ReLU', 
                                layer_args=dict(inplace=False)), 
                 
                 adapter_layernorm_option="in",
                 norm_cfg = dict(norm_type='LayerNorm', 
                     requires_grad=True,
                     ),
                 
                 dropout_layer = dict(drop_type='Dropout',
                                        drop_prob=0.0,
                                        inplace=False),
                 
                 
                 ):
        super().__init__()
        self.n_embed = in_channels  
        self.down_size = bottleneck
        
        # the position of norm
        self.adapter_layernorm_option = adapter_layernorm_option
    
        
        
        if adapter_layernorm_option == 'in' or adapter_layernorm_option == 'out':
            self.adapter_layer_norm = build_norm_layer(num_features=in_channels, **norm_cfg)[1]
        else: 
            self.adapter_layer_norm = None
        
        
        self.down_proj = nn.Linear(self.n_embed, self.down_size)
        self.non_linear_func = build_activation_layer(**act_cfg)
        self.up_proj = nn.Linear(self.down_size, self.n_embed)
        self.dropout = build_dropout(**dropout_layer)
        
        # initalize weights
        self.init_weights()
        
        
    def init_weights(self):
        kaiming_init(self.down_proj, bias=0, distribution='uniform')
        constant_init(self.up_proj, 0, bias=0)


    def forward(self, x, add_residual=False, residual=None):
        residual = x if residual is None else residual
        
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm(x)
        
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        down = self.dropout(down)
        up = self.up_proj(down)
        
        # up = up * self.scale
        
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)
        
        if add_residual:
            up = up + residual
        
        return up



class Compensator(nn.Module):
    '''
    Args:
        in_channels: input channels $d$ e.g. C
        bottleneck (int): $\hat{d}$
        
        
        adapter_scalar (str): s. Could be learnable scalar i.e. 'learnable_scalar'
            Default: '1.0'
            
        act_cfg: activation function config. See src.Models.utils.
            Default: dict(
                act_type = 'ReLU', 
                inplace = False
            )
            
        adapter_layernorm_option: The position of norm layer.
            'in' for the norm before the down layer.
            'out' for the norm after the up layer.
            'none': no norm layer
            Default: 'in'
            
        norm_cfg: norm config.
            Default: dict(norm_type='LayerNorm', 
                                 requires_grad=True,
                                 eps=1e-5,
                                 affine=True)
            
        dropout_layer: dropout config
            Default: dict(drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
        
    Shape:
        In: (N, L, C)
        Out: (N, L, C)
    
    '''
    def __init__(self, 
                 in_channels, 
                 
                 # finetune cfg
                 bottleneck: list = None, 
                  
                 pt_act_cfg = dict(act_type='ReLU', 
                                layer_args=dict(inplace=False)),
                 
                 adapter_layernorm_option="in",
                 norm_cfg = dict(norm_type='LayerNorm', 
                     requires_grad=True,
                     ),
                 
                 dropout_layer = dict(drop_type='Dropout',
                                        drop_prob=0.0,
                                        inplace=False),
                 
                 
                 ):
        super().__init__()
        self.n_embed = in_channels  
        self.pt_down_size = bottleneck
        
        # the position of norm
        self.adapter_layernorm_option = adapter_layernorm_option
    
        
        
        if adapter_layernorm_option == 'in' or adapter_layernorm_option == 'out':
            self.adapter_layer_norm = build_norm_layer(num_features=in_channels, **norm_cfg)[1]
        else: 
            self.adapter_layer_norm = None
        
            
            
        
        
        self.down_conv1 = nn.Conv2d(in_channels=self.n_embed, out_channels=self.pt_down_size, kernel_size=1)
        self.down_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=self.pt_down_size, out_channels=self.pt_down_size, kernel_size=3, padding=1, groups=self.pt_down_size), 
            nn.Conv2d(in_channels=self.pt_down_size, out_channels=self.pt_down_size, kernel_size=1)
        )
        
        self.conv_actfunc = build_activation_layer(**pt_act_cfg)
        self.up_conv = nn.Conv2d(in_channels=self.pt_down_size, out_channels=self.n_embed, kernel_size=1)
        
        
        # initalize weights
        self.init_weights()
        
        
    def init_weights(self):
        kaiming_init(self.down_conv1, bias=0, distribution='uniform')
        kaiming_init(self.down_conv2, bias=0, distribution='uniform')
        constant_init(self.up_conv, 0, bias=0)


    def forward(self, x, p_token_start_idx, add_residual=False, residual=None):
        residual = x if residual is None else residual
        B, L, C = x.shape
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm(x)
        
        
        
        p_token = x[:, p_token_start_idx:, :] 
        h = int(math.sqrt(L - p_token_start_idx))
        w = h
        # p_token = self.norm(p_token)
        p_token = rearrange(p_token, 'b (h w) c -> b c h w', h=h, w=w)
        
        pt_down = self.down_conv1(p_token)
        pt_down = self.down_conv2(pt_down)
        pt_down = self.conv_actfunc(pt_down)
        pt_up = self.up_conv(pt_down)
        
        pt_up = rearrange(pt_up, 'b c h w -> b (h w) c')
        
        up = pt_up
        
        if self.adapter_layernorm_option == 'out':
            up = self.adapter_layer_norm(up)
        
        if add_residual:
            up = up + residual
        
        return up

