from functools import partial
import logging
import math
import os
import warnings

from typing import (Sequence, Tuple, Union, 
                    Callable, Optional, Dict, 
                    Any, List)

import torch
import torch.nn as nn
from torch.nn.modules import GELU, LayerNorm, Module
import torch.utils.checkpoint
import torch.nn.functional as F

import einops

from src.Models.backbones.dinov2_vit_l import Attention, Block, Mlp, PatchEmbed


from .dinov2_vit_l import (Block, MemEffAttention, drop_add_residual_stochastic_depth, drop_add_residual_stochastic_depth_list, 
                           LayerScale, get_attn_bias_and_cat, DinoVisionTransformer, BlockChunk, SwiGLUFFNFused)
from .dinov2_vit_l import XFORMERS_AVAILABLE

from ..finetunes.msk_gen import TokenSelect

logger = logging.getLogger("dinov2_dar")

        
        
from ..finetunes.dar import Adapter, Compensator

class Block_DAR(Block):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        proj_bias: bool = True,
        ffn_bias: bool = True,
        drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values=None,
        drop_path: float = 0.0,
        act_layer: Callable[..., nn.Module] = nn.GELU,
        norm_layer: Callable[..., nn.Module] = nn.LayerNorm,
        attn_class: Callable[..., nn.Module] = Attention,
        ffn_layer: Callable[..., nn.Module] = Mlp,
        ft_cfg = dict(
                    bottleneck=[72, 48], 
                    adapter_scalar=0.1, 
                    
                    act_cfg=dict(
                        act_type='ReLU', 
                        layer_args=dict(inplace=False)
                        ),   
                    pt_act_cfg=dict(
                        act_type='ReLU', 
                        layer_args=dict(inplace=False)
                        ),      
                    adapter_layernorm_option='none',
                            
                    dropout_layer = dict(
                        drop_type='Dropout',
                        drop_prob=0.0,
                        inplace=False)
                ),
        p_token_start_idx=1,
        is_select: bool = False,
        
    ) -> None:
        super().__init__(dim, num_heads, mlp_ratio, qkv_bias, proj_bias, ffn_bias, drop, attn_drop, init_values, drop_path, act_layer, norm_layer, attn_class, ffn_layer)

        adapter_cfg = dict(
            bottleneck=ft_cfg['bottleneck'][0],
            act_cfg=ft_cfg['act_cfg'],
            adapter_layernorm_option=ft_cfg['adapter_layernorm_option'],
            dropout_layer=ft_cfg['dropout_layer']
        )
        self.adapter = Adapter(in_channels=dim, **adapter_cfg)
        compstr_cfg = dict(
            bottleneck=ft_cfg['bottleneck'][1],
            pt_act_cfg=ft_cfg['pt_act_cfg'],
            adapter_layernorm_option=ft_cfg['adapter_layernorm_option'],
            dropout_layer=ft_cfg['dropout_layer']
        )
        self.compensator = Compensator(in_channels=dim, **compstr_cfg)
        self.scale = nn.Parameter(torch.tensor(ft_cfg['adapter_scalar']))
        
        
        self.is_select = is_select
        if self.is_select:
            self.token_mask_gen = TokenSelect(dim, num_sub_layer=1, p_token_start_idx=p_token_start_idx)
        else:
            self.token_mask_gen = None
            
        self.p_token_start_idx = p_token_start_idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls1(self.attn(self.norm1(x)))

        def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
            return self.ls2(self.mlp(self.norm2(x)))
        
        x = x + attn_residual_func(x)
        
        # ======================================================================================
        if self.training:
            
            
            policy_token = x
            token_mask, token_logits = self.token_mask_gen(policy_token)
            mlp_x = ffn_residual_func(x)
            if token_mask is not None:
                mlp_x = token_mask * mlp_x
            
            adpt_x = self.adapter(x)
            adpt_x[:, self.p_token_start_idx:, :] = adpt_x[:, self.p_token_start_idx:, :] + self.compensator(x, p_token_start_idx=self.p_token_start_idx)
            adpt_x = adpt_x * self.scale
            
            x = x + mlp_x + adpt_x
        # else:
        #     policy_token = x
        #     sub_token_select, token_logits = self.mlp_token_select(policy_token)
            
        #     adpt_x = self.adapter(x, p_token_start_idx=self.p_token_start_idx)
        #     if sub_token_select is not None:
        #         mlp_x = ffn_residual_func(x * sub_token_select)
        #         mlp_x = mlp_x * sub_token_select
        #     else:
        #         mlp_x = ffn_residual_func(x)
        else:
            policy_token = x
            token_mask, token_logits = self.token_mask_gen(policy_token)
            
            adpt_x = self.adapter(x)
            adpt_x[:, self.p_token_start_idx:, :] = adpt_x[:, self.p_token_start_idx:, :] + self.compensator(x, p_token_start_idx=self.p_token_start_idx)
            adpt_x = adpt_x * self.scale
            
            if token_mask is not None:
                idx = torch.nonzero(token_mask.detach(), as_tuple=True)[1] # (K, )
                idx = idx.unsqueeze(0) # (1, K, )
                idx = einops.repeat(idx, 'b k -> b k d', d=self.dim)
                mlp_x = torch.gather(x, dim=1, index=idx)
                mlp_x = ffn_residual_func(mlp_x)
                x = x + adpt_x.scatter_add(dim=1, index=idx, src=mlp_x) 
                
            else:
                mlp_x = ffn_residual_func(x)
                x = x + mlp_x + adpt_x
       # ==================================================================================
        
       
        return x, dict(sub_token_select=token_mask, token_logits=token_logits)
                    
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     def attn_residual_func(x: torch.Tensor) -> torch.Tensor:
    #         return self.ls1(self.attn(self.norm1(x)))

    #     def ffn_residual_func(x: torch.Tensor) -> torch.Tensor:
    #         return self.ls2(self.mlp(self.norm2(x)))
        
    #     x = x + attn_residual_func(x)
        
    #     policy_token = x
    #     sub_token_select, token_logits = self.mlp_token_select(policy_token)
        
    #     adpt_x = self.adapter(x)
    #     if sub_token_select is not None:
    #         mlp_x = ffn_residual_func(x * sub_token_select)
    #         mlp_x = mlp_x * sub_token_select.detach()
    #     else:
    #         mlp_x = ffn_residual_func(x)
       
    #     x = x + mlp_x + adpt_x
            
        
    #     return x, dict(sub_token_select=sub_token_select, token_logits=token_logits)



class DinoVisionTransformer_DAR(DinoVisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=True,
        ffn_bias=True,
        proj_bias=True,
        drop_path_rate=0.0,
        drop_path_uniform=False,
        init_values=None,  # for layerscale: None or 0 => no layerscale
        embed_layer=PatchEmbed,
        act_layer=nn.GELU,
        block_fn=partial(Block_DAR, attn_class=MemEffAttention),
        ffn_layer="mlp",
        block_chunks=1,
        num_register_tokens=0,
        interpolate_antialias=False,
        interpolate_offset=0.1,
        
        ft_cfg=dict(
            
        ),
    ):
        super(DinoVisionTransformer, self).__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 1
        self.n_blocks = depth
        self.num_heads = num_heads
        self.patch_size = patch_size
        self.num_register_tokens = num_register_tokens
        self.interpolate_antialias = interpolate_antialias
        self.interpolate_offset = interpolate_offset
        

        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        assert num_register_tokens >= 0
        self.register_tokens = (
            nn.Parameter(torch.zeros(1, num_register_tokens, embed_dim)) if num_register_tokens else None
        )

        if drop_path_uniform is True:
            dpr = [drop_path_rate] * depth
        else:
            dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        if ffn_layer == "mlp":
            logger.info("using MLP layer as FFN")
            ffn_layer = Mlp
        elif ffn_layer == "swiglufused" or ffn_layer == "swiglu":
            logger.info("using SwiGLU layer as FFN")
            ffn_layer = SwiGLUFFNFused
        elif ffn_layer == "identity":
            logger.info("using Identity layer as FFN")

            def f(*args, **kwargs):
                return nn.Identity()

            ffn_layer = f
        else:
            raise NotImplementedError

        blocks_list = [
            block_fn(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                proj_bias=proj_bias,
                ffn_bias=ffn_bias,
                drop_path=dpr[i],
                norm_layer=norm_layer,
                act_layer=act_layer,
                ffn_layer=ffn_layer,
                init_values=init_values,
                ft_cfg=ft_cfg,
                p_token_start_idx=self.num_tokens + self.num_register_tokens,
                is_select=True,
            )
            for i in range(depth)
        ]
        if block_chunks > 0:
            self.chunked_blocks = True
            chunked_blocks = []
            chunksize = depth // block_chunks
            for i in range(0, depth, chunksize):
                # this is to keep the block index consistent if we chunk the block list
                chunked_blocks.append([nn.Identity()] * i + blocks_list[i : i + chunksize])
            self.blocks = nn.ModuleList([BlockChunk(p) for p in chunked_blocks])
        else:
            self.chunked_blocks = False
            self.blocks = nn.ModuleList(blocks_list)

        self.norm = norm_layer(embed_dim)
        # self.head = nn.Identity()

        # self.mask_token = nn.Parameter(torch.zeros(1, embed_dim))
        
        
      
        
    def forward_features(self, x):
        
        # if isinstance(x, list):
        #     return self.forward_features_list(x)

        x = self.prepare_tokens_with_masks(x, masks=None)

        token_select_list = []
        token_logits_list = []
        for i, blk in enumerate(self.blocks):
            
            x, token_select_dict = blk(x)
            if (token_select_dict["sub_token_select"] is not None) and (token_select_dict["token_logits"] is not None):
                token_select_list.append(token_select_dict["sub_token_select"])
                token_logits_list.append(token_select_dict["token_logits"])
           
        token_select = self.convert_list_to_tensor(token_select_list)[:, :, self.num_tokens + self.num_register_tokens:, :] # remove cls token
        token_logits = self.convert_list_to_tensor(token_logits_list)
        

        x_norm = self.norm(x)
        
        
        
        return {
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_regtokens": x_norm[:, 1 : self.num_register_tokens + 1],
            "x_norm_patchtokens": x_norm[:, self.num_register_tokens + 1 :],
            "x_prenorm": x,
            
        }, dict(token_select=token_select, token_logits=token_logits)
        
        
    @staticmethod
    def convert_list_to_tensor(list_convert):
        if len(list_convert):
            result = torch.stack(list_convert, dim=1)
        else :
            result = None
        return result 
    
    
    def get_attn_weights_layers(self, x:torch.Tensor, n=12):
        x = self.prepare_tokens_with_masks(x, masks=None)
        token_select_list = []
        token_logits_list = []
        output, i, total_block_len = [], 0, len(self.blocks)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            attn_weights = blk.get_attn_weights(x)
            
            x, token_select_dict = blk(x)
            if isinstance(x, tuple):
                x = x[0]
            
            if i in blocks_to_take:
                output.append(attn_weights)
        assert len(output) == len(blocks_to_take), f"only {len(output)} / {len(blocks_to_take)} blocks found"
        return output
    
    
    def _get_intermediate_layers_not_chunked(self, x, n=1):
        x = self.prepare_tokens_with_masks(x)
        # If n is an int, take the n last blocks. If it's a list, take them
        output, total_block_len = [], len(self.blocks)
        output.append(x)
        blocks_to_take = range(total_block_len - n, total_block_len) if isinstance(n, int) else n
        for i, blk in enumerate(self.blocks):
            x, token_select_dict = blk(x)
            if i in blocks_to_take:
                output.append(x)
        assert len(output)-1 == len(blocks_to_take), f"only {len(output)-1} / {len(blocks_to_take)} blocks found"
        return output
        
        
    def get_intermediate_layers(
        self,
        x: torch.Tensor,
        n: Union[int, Sequence] = 1,  # Layers or n last layers to take
        reshape: bool = False,
        return_class_token: bool = False,
        norm=True,
    ) -> Tuple[Union[torch.Tensor, Tuple[torch.Tensor]]]:
        if self.chunked_blocks:
            outputs = self._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self._get_intermediate_layers_not_chunked(x, n)
        if norm:
            outputs = [self.norm(out) for out in outputs]
        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, 1 + self.num_register_tokens :] for out in outputs]
        if reshape:
            B, _, w, h = x.shape
            outputs = [
                out.reshape(B, w // self.patch_size, h // self.patch_size, -1).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)
    
    
    