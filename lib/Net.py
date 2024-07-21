"""
@File: Net.py
@Time: 2023/11/30
@Author: rp
@Software: PyCharm

"""
import torch
import torch.nn.functional as F

from thop import profile, clever_format
from torch import nn as nn
from lib.SMT import smt_t,smt_s
from lib.modules import Depth_Backbone, DimensionalReduction, Fusion, LightDecoder, MaskGuidance, CMFM, MFFM, Block,Refine


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

        self.rgb_encoder = smt_t(pretrained=True)
        self.depth_encoder = Depth_Backbone()
        channel_list = [64, 128, 256, 512]

        self.dr1 = DimensionalReduction(channel_list[1], 64)
        self.dr2 = DimensionalReduction(channel_list[2], 64)
        self.dr3 = DimensionalReduction(channel_list[3], 64)


        self.fusion1 = MFFM(dim=64)
        self.fusion2 = MFFM(dim=64)
        self.fusion3 = MFFM(dim=64)
        self.fusion4 = MFFM(dim=64)


        self.lightdecoder = LightDecoder(channel=64)
        self.refine = Refine()
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.up8 = nn.UpsamplingBilinear2d(scale_factor=8)
        self.up16 = nn.UpsamplingBilinear2d(scale_factor=16)
        self.up32 = nn.UpsamplingBilinear2d(scale_factor=32)
        self.mg = MaskGuidance(channel=64, group_num=32)
        self.mg1 = MaskGuidance(channel=64, group_num=32)
        self.mg2 = MaskGuidance(channel=64, group_num=32)
        self.mg3 = MaskGuidance(channel=64, group_num=32)


        self.block1 = Block(128)
        self.block2 = Block(192)
        self.block3 = Block(256)

        self.salhead1 = nn.Conv2d(in_channels=64, out_channels=1, kernel_size=3, padding=1)
        self.salhead2 = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, padding=1)
        self.salhead3 = nn.Conv2d(in_channels=192, out_channels=1, kernel_size=3, padding=1)
        self.salhead4 = nn.Conv2d(in_channels=256, out_channels=1, kernel_size=3, padding=1)

    def forward(self, r, d):
        rgb = self.rgb_encoder(r)
        d = self.depth_encoder(d)

        rgb1 = rgb[0]
        rgb2 = self.dr1(rgb[1])
        rgb3 = self.dr2(rgb[2])
        rgb4 = self.dr3(rgb[3])

        d1 = d[0]
        d2 = self.dr1(d[1])
        d3 = self.dr2(d[2])
        d4 = self.dr3(d[3])


        f1 = self.fusion1(rgb1, d1)
        f2 = self.fusion2(rgb2, d2)
        f3 = self.fusion3(rgb3, d3)
        f4 = self.fusion4(rgb4, d4)

        init_pred = self.lightdecoder(f4, f3, f2)
        init_pred1 = self.up8(init_pred)  # 粗略显著图

        f4_3 = F.interpolate(f4, size=f3.size()[2:], mode='bilinear', align_corners=True)
        f3_2 = F.interpolate(f3, size=f2.size()[2:], mode='bilinear', align_corners=True)
        f2_1 = F.interpolate(f2, size=f1.size()[2:], mode='bilinear', align_corners=True)
        f1_m = self.mg((f2_1 + f1), init_pred)
        f2_m = self.mg1((f3_2 + f2), init_pred)  # [1,64,44,44]
        f3_m = self.mg2((f4_3 + f3), init_pred)  # [1,64,22,22]
        # f4_m = self.mg3(f4, init_pred)# [1,64,11,11]
        f4_m = self.refine(init_pred, f4)

        f4_m_pred = self.salhead1(self.up32(f4_m))  # [1,1,352,352]
        f4_3_m = torch.cat([f3_m, self.up2(f4_m)], dim=1)  # [1,128,22,22]
        f4_3_m = self.block1(f4_3_m)
        f4_3_m_pred = self.salhead2(self.up16(f4_3_m))  # [1,1, 352,352]

        f4_3_2_m = torch.cat([f2_m, self.up2(f4_3_m)], dim=1)  # [1,192,44,44]
        f4_3_2_m = self.block2(f4_3_2_m)
        f4_3_2_m_pred = self.salhead3(self.up8(f4_3_2_m))  # [1,1, 352,352]
        f4_3_2_1_m = torch.cat([f1_m, self.up2(f4_3_2_m)], dim=1)  # [1,256,88,88]
        f4_3_2_1_m = self.block3(f4_3_2_1_m)
        f4_3_2_1_m_pred = self.salhead4(self.up4(f4_3_2_1_m))

        return f4_3_2_1_m_pred, f4_3_2_m_pred, f4_3_m_pred, f4_m_pred, init_pred1


if __name__ == '__main__':
    model = Net()
    r = torch.randn([1, 3, 384, 384])
    d = torch.randn([1, 3, 384, 384])
    # print(model)
    params, flops = profile(model, inputs=(r, d))
    flops, params = clever_format([flops, params], "%.2f")
    print(flops, params)
