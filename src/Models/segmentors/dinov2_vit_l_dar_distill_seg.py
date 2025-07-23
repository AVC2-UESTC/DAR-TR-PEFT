import warnings
import math

from copy import deepcopy

import torch
import torch.nn as nn
# import torch.nn.functional as F
from typing import List, Tuple, Dict
from torch import Tensor

from einops import rearrange

# from ..builder import build_loss

from ..utils import resize

from ..segmentors import BaseSegmenter, BaseSegmentor_Config


from ..backbones.dinov2_vit_l_ft_dar import DinoVisionTransformer_DAR, DinoVisionTransformer
from ..decode_heads.simp_decoder import SimpleDecoder, DAR_ditill_head

class DinoVisionTransformer_DAR_distill_Seg(BaseSegmenter):
    '''
    Args:

    '''
    def __init__(self, 
                 backbone_t_cfg: Dict, 
                 backbone_s_cfg: Dict,
                 decode_head_cfg: Dict, 
                 d_head_cfg: Dict,
                 
                 threshold: float = None, 
                 loss_decode: Dict = dict(loss_type='CrossEntropyLoss', 
                                  reduction='mean'),  
                 ignore_index: int = 255, 
                 align_corners: bool = False) -> None:
        super().__init__(threshold=threshold, loss_decode=loss_decode, ignore_index=ignore_index, align_corners=align_corners)
        
        
        self.backbone = DinoVisionTransformer_DAR(**backbone_s_cfg)

        self.decode_head = SimpleDecoder(**decode_head_cfg)
        
        if self.training:
            self.backbone_t = DinoVisionTransformer(**backbone_t_cfg)
            self.d_head_t = DAR_ditill_head(**d_head_cfg)
            self.d_head_s = DAR_ditill_head(**d_head_cfg)
        else:
            self.backbone_t = None
            self.d_head_t = None
            self.d_head_s = None

        out_channels = decode_head_cfg['out_channels']
        
        if out_channels == 1 and threshold is None:
            # threshold = 0.3
            warnings.warn('threshold is not defined for binary')
            
        self.patch_size = self.backbone.patch_size
        
        self.p_token_start_idx = self.backbone.num_register_tokens + 1
        self.depth = self.backbone.n_blocks

    
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
        
        if self.d_head_t is not None:
            x_rec = self.d_head_s(patchtokens) # (B, L, C)
            
            out_dict_t = self.backbone_t(x)
            rec_tgt = out_dict_t['x_norm_patchtokens']
            # rec_tgt = rearrange(rec_tgt, 'b (h w) c -> b c h w', h=h, w=w)
            rec_tgt = self.d_head_t(rec_tgt) # (B, L, C)
            
        patchtokens = rearrange(patchtokens, 'b (h w) c -> b c h w', h=h, w=w)
        patchtokens = [patchtokens, ]

        results = self.decode_head(patchtokens)
        results = dict(**results, **token_select_dict)
        # {'logits_mask':..., 'token_select':..., 'token_logits':...}
        
        if self.d_head_t is not None:
            # mim_mask = token_select_dict['token_select'] # (B, depth,  L, 1)
            # mim_mask = mim_mask[:, self.depth - 1, :, :] # (B, L, 1)
            # # mim_mask = mim_mask[:, self.p_token_start_idx:, :]
            # mim_mask = 1 - mim_mask
            # results['mim_mask'] = mim_mask
            results['reconstruction'] = x_rec
            results['rec_target'] = rec_tgt
        
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
        
        rec_target = results['rec_target']
        rec = results['reconstruction']
        # mim_mask = results['mim_mask']
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
            elif loss_decode.loss_name.startswith('d_reg_'):
                if loss_decode.loss_name not in losses:
                    losses[loss_decode.loss_name] = loss_decode(
                        rec,
                        rec_target,#(N, H, W)
                        extra_in = None
                        )
                else:
                    losses[loss_decode.loss_name] += loss_decode(
                        rec,
                        rec_target,
                        extra_in = None
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
    



class DinoVisionTransformer_DAR_distill_Seg_Config(BaseSegmentor_Config):
    '''
    
    '''
    def __init__(self, 
                 pretrained_weights: str = None, 
                 finetune_weights: str = None, 
                 tuning_mode: str = 'PEFT', 
                 
                backbone_t_cfg: Dict = None, 
                backbone_s_cfg: Dict = None,
                decode_head_cfg: dict = None,
                d_head_cfg: dict = None,
                
                threshold = None, 
                loss_decode=dict(
                     loss_type='CrossEntropyLoss',
                     reduction = 'mean',
                     ),
                ignore_index=255,
                align_corners: bool = False) -> None:
        super().__init__(pretrained_weights, finetune_weights, tuning_mode, threshold, loss_decode, ignore_index, align_corners)        
        self.backbone_t_cfg = backbone_t_cfg
        self.backbone_s_cfg = backbone_s_cfg
        self.decode_head_cfg = decode_head_cfg
        self.d_head_cfg = d_head_cfg
        
        
    def set_model_class(self):
        self.model_class = DinoVisionTransformer_DAR_distill_Seg


    @property
    def model(self):
        self.set_model_class()
        model_args = deepcopy(self.__dict__)
        model_args.pop('pretrained_weights')
        model_args.pop('finetune_weights')
        model_args.pop('tuning_mode')
        model_args.pop('model_class')
        model_inst = self.model_class(**model_args)
        
        
        if self.pretrained_weights is not None:
            print('Pretrained weights loaded.')
            model_weights = torch.load(self.pretrained_weights, map_location='cpu')
            teacher_model_weights = dict()
            for k, v in model_weights.items():
                new_name = k.replace('backbone.', 'backbone_t.')
                teacher_model_weights[new_name] = v 
                
            # for k, v in model_inst.state_dict().items():
            #     print(k)
            
            # for k, v in model_weights.items():
            #     try:
            #         print(k)
            #         print(model_inst.state_dict()[k].shape)
            #     except KeyError:
            #         print(f'{k} not in model state dict')
            #         for k2, v2 in model_inst.state_dict().items():
            #             print(k2)
            #         break
            
            model_weights = {**teacher_model_weights, **model_weights}
            
            load_weights_dict = {k: v for k, v in model_weights.items()
                                if model_inst.state_dict()[k].numel() == v.numel()}
            
            msg = model_inst.load_state_dict(load_weights_dict, strict=False)
            
            
            if self.tuning_mode == 'PEFT':
                print('Start Parameter-Efficient Finetuning')
                weights_tosave_keys = msg.missing_keys
                
                if self.finetune_weights is not None:
                    print('Finetune weights loaded')
                    model_finetune_weights = torch.load(self.finetune_weights, map_location='cpu')
                    load_weights_dict = {k: v for k, v in model_finetune_weights.items()
                                 if model_inst.state_dict()[k].numel() == v.numel()}
                    model_inst.load_state_dict(load_weights_dict, strict=False)
                    
                # freeze all but finetune and head
                for name, p in model_inst.named_parameters():
                    if name in weights_tosave_keys:
                        p.requires_grad = True
                    else:
                        p.requires_grad = False
                    
            elif self.tuning_mode == 'Full':
                print('Start Full Finetuning')
                weights_tosave_keys = model_inst.state_dict().keys()
                
                if self.finetune_weights is not None:
                    print('Finetune weights loaded')
                    model_finetune_weights = torch.load(self.finetune_weights, map_location='cpu')
                    load_weights_dict = {k: v for k, v in model_finetune_weights.items()
                                 if model_inst.state_dict()[k].numel() == v.numel()}
                    model_inst.load_state_dict(load_weights_dict, strict=False)
                
        else:
            print('No pretrained weights. Start training from scratch.')
            weights_tosave_keys = model_inst.state_dict().keys()
        
        return model_inst, weights_tosave_keys




