from torch import nn
from .conv_blocks import DepthWiseSeperableConv, get_norm
from .hybrid_block import HybridBlock

# # 4. DHVN Encoder
class DHVNEncoder(nn.Module):
    def __init__(self, in_ch=3, base_dim=32, norm="group"):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base_dim*2, kernel_size=3, padding=1, bias=False),
            get_norm(norm, base_dim*2),
            nn.ReLU(inplace=True)
        )
        self.s1 = DepthWiseSeperableConv(base_dim*2, base_dim*2, norm=norm)
        self.s2 = HybridBlock(base_dim*2, base_dim*4, stage=2, transformer_depth=4, norm=norm)
        self.s3 = HybridBlock(base_dim*4, base_dim*8, stage=3, transformer_depth=6, norm=norm)
        self.bottleneck = HybridBlock(base_dim*8, base_dim*16, stage=0, transformer_depth=8, norm=norm)
    
    def forward(self, x):
        x = self.stem(x)
        s1_out = self.s1(x)
        s2_out, gate2 = self.s2(s1_out)
        s3_out, gate3 = self.s3(s2_out)
        btl_out, btl_gate = self.bottleneck(s3_out)

        feats_out = {"d_s1":s1_out, "d_s2":s2_out, "d_s3":s3_out, "bottleneck": btl_out}
        gates = {"d_g2":gate2, "d_g3":gate3, "bottleneck":btl_gate}
        
        return feats_out, gates
        # pass
