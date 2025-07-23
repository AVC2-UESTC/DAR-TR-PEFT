
import torch 
import torch.nn as nn

from einops import rearrange

from mmengine.model import kaiming_init, constant_init


def _gumbel_sigmoid(
    logits, tau=1, hard=False, eps=1e-10, training = True, threshold = 0.5
):
    if training :
        # ~Gumbel(0,1)`
        gumbels1 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        gumbels2 = (
            -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format)
            .exponential_()
            .log()
        )
        # Difference of two` gumbels because we apply a sigmoid
        gumbels1 = (logits + gumbels1 - gumbels2) / tau
        y_soft = gumbels1.sigmoid()
    else :
        y_soft = logits.sigmoid()

    if hard:
        # Straight through.
        y_hard = torch.zeros_like(
            logits, memory_format=torch.legacy_contiguous_format
        ).masked_fill(y_soft > threshold, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        ret = y_soft
    return ret


    
    

    
class TokenSelect(nn.Module):
    def __init__(self, dim_in, num_sub_layer, p_token_start_idx, tau=5, is_hard=True, threshold=0.5, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(dim_in, num_sub_layer, kernel_size=3, padding=1, bias=bias)
        # nn.Linear(dim_in, num_sub_layer, bias=bias)

        self.p_token_start_idx = p_token_start_idx

        self.is_hard = is_hard
        self.tau = tau
        self.threshold = threshold

    def forward(self, x):
        b, l = x.shape[:2]
        
        h = w = int((l - self.p_token_start_idx) ** 0.5)
        logits = rearrange(x[:, self.p_token_start_idx:, :], 'b (h w) c -> b c h w', h=h, w=w)
        
        logits = self.conv(logits)
        logits = rearrange(logits, 'b c h w -> b (h w) c')
        
        token_mask = _gumbel_sigmoid(logits, self.tau, self.is_hard, threshold=self.threshold, training=self.training)
        token_mask = torch.cat([token_mask.new_ones(b, self.p_token_start_idx, 1), token_mask], dim=1)
        
        return token_mask, logits
    
    
