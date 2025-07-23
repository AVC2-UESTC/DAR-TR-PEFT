import warnings
import math

from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor

from einops import rearrange

from ..builder import build_loss

from ..utils import resize

from ..segmentors import BaseSegmenter, BaseSegmentor_Config



from ..decode_heads.simp_decoder import SimpleDecoder
from ..backbones.dinov2_vit_l_ft_dar import DinoVisionTransformer_DAR 


class DinoVisionTransformer_DAR_Seg(BaseSegmenter):
    '''
    Args:

    '''
    def __init__(self, 
                 backbone_cfg: Dict, 
                 decode_head_cfg: Dict, 
                 
                 threshold: float = None, 
                 loss_decode: Dict = dict(loss_type='CrossEntropyLoss', 
                                  reduction='mean'),  
                 ignore_index: int = 255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold=threshold, loss_decode=loss_decode, ignore_index=ignore_index, align_corners=align_corners)
        
        self.backbone = DinoVisionTransformer_DAR(**backbone_cfg)

        self.decode_head = SimpleDecoder(**decode_head_cfg)

        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')

    
    def forward(self, inputs: Dict):
        
        x = inputs['image']
        
        out_dict, token_select_dict = self.backbone(x)
        # e.g.: x_norm_clstoken torch.Size([1, 1024])
        #       x_norm_regtokens torch.Size([1, 4, 1024])
        #       x_norm_patchtokens torch.Size([1, 256, 1024])
        #       x_prenorm torch.Size([1, 261, 1024])
        patchtokens = out_dict['x_norm_patchtokens']
        N, L, C = patchtokens.size()
        h = w = int(L ** 0.5)
        patchtokens = rearrange(patchtokens, 'b (h w) c -> b c h w', h=h, w=w)
        patchtokens = [patchtokens, ]

        results = self.decode_head(patchtokens)
        results = dict(**results, **token_select_dict)
        # {'logits_mask':..., 'token_select':..., 'token_logits':...}
        return results
    
    
    def loss(self, inputs: Dict[str, Tensor], labels: Dict[str, Tensor],
             return_logits: bool = False
             ) -> dict:
        """Forward function for training.

        Args:
            

        Returns:
            
        Shape:
            inputs: dict(
                image: (N, C, H, W)
            )
            
            labels: dict(
                label_mask: (N, out_channel, H, W)
            ) 
            
        """
        results = self.forward(inputs)
        
        token_select = results['token_select']
        # token_logits = results['token_logits']
        
        seg_logits = results['logits_mask']
        
        seg_label = labels['label_mask']
        
        logits_prob = torch.sigmoid(seg_logits) # for metric computing
        
        
        seg_logits = resize(
            input=seg_logits,
            size=seg_label.shape[2:], #(N, 1, H, W)
            mode='bilinear',
            align_corners=self.align_corners)
        
        seg_label = seg_label.squeeze(1)
        # (N, H, W)
        
        # Calculate loss
        losses = dict()
        
        if not isinstance(self.loss_decode, nn.ModuleList):
            losses_decode = [self.loss_decode]
            
        else:
            losses_decode = self.loss_decode
        # losses_decode: loss layer(s) in Modulelist
        
        for loss_decode in losses_decode:
            if loss_decode.loss_name.startswith('mask_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        seg_logits,
                        seg_label,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        seg_logits,
                        seg_label,
                        )
            elif loss_decode.loss_name.startswith('tokenreg_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        token_select,
                        token_select,#(N, H, W)
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        token_select,
                        token_select,
                        )
            else:
                raise ValueError(f'loss name: {loss_decode.loss_name} is not supported')
                    
               
        # losses: {
        #         
        #         'loss_name1': loss_value1
        #         ...
        #     }
        
        preds = dict(pred_mask=logits_prob)
        
        if return_logits:
            return losses, preds
        else:
            return losses
    



class DinoVisionTransformer_DAR_Seg_Config(BaseSegmentor_Config):
    '''
    
    '''
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                backbone_cfg: dict = None, 
                decode_head_cfg: dict = None,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)        
        self.backbone_cfg = backbone_cfg
        self.decode_head_cfg = decode_head_cfg
        
        
    def set_model_class(self):
        self.model_class = DinoVisionTransformer_DAR_Seg
