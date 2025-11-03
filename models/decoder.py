import torch
from torch import nn
import torch.nn.functional as F
from .hybrid_block import HybridBlock

# # 5. DHVN Decoder

# ## 5.1. UpSample Block
class UpSample(nn.Module):
    def __init__(self, in_ch, out_ch, skip_ch=None, stage=3, transformer_depth=2, norm="group"):
        super().__init__()
        if skip_ch is not None:
            self.hybrid = HybridBlock(in_ch + skip_ch, out_ch, stage, transformer_depth, norm=norm)
        else: 
            self.hybrid = HybridBlock(in_ch, out_ch, stage, transformer_depth, norm=norm)

        self.identity = nn.Identity()
    
    def forward(self, x, skip=None):
        if skip is not None:
            h, w = skip.shape[-2]*2, skip.shape[-1]*2
            x = F.interpolate(x, (h,w), mode="bilinear", align_corners=False)
            skip = F.interpolate(skip, (h,w), mode="bilinear", align_corners=False)
            x = torch.cat([x, skip], dim=1)
            x, gate = self.hybrid(x)
        else:
            h, w = x.shape[-2]*4, x.shape[-1]*4
            x = F.interpolate(x, (h,w), mode="bilinear", align_corners=False)
            x, gate = self.hybrid(x)
        return x, gate


# ## 5.2. Decoder block Imp
class DHVNDecoder(nn.Module):
    def __init__(self, base_dim=32, norm="group"):
        super().__init__()
        self.s3 = UpSample(base_dim*16, base_dim*8, base_dim*8, stage=3, transformer_depth=2, norm=norm)
        self.s2 = UpSample(base_dim*8, base_dim*4, base_dim*4, stage=2, transformer_depth=2, norm=norm)
        self.s1 = UpSample(base_dim*4, base_dim*2, base_dim*2, norm=norm)

    def forward(self, feats, classification=True):
        s3, g3 = self.s3(feats['bottleneck'], feats['d_s3'])
        s2, g2 = self.s2(s3, feats['d_s2'])
        out_feats = {"u_s3": s3, "u_s2": s2}
        out_gates = {"u_g3": g3, "u_g2": g2}
        if not classification:
            s1, g1 = self.s1(s2, feats['d_s1'])
            out_feats["u_s1"] = s1
            out_gates["u_g1"] = g1      

        return out_feats, out_gates
