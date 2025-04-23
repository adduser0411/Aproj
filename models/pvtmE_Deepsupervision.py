


import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbone.pvtv2 import pvt_v2_b2
from models.blocks.ACmixMerge import ACmix
from models.blocks.Mambaout import GatedCNNBlock
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout
from torch.jit import Final
from functools import partial
from typing import List, Callable
from torch import Tensor
import os
from timm.layers import DropPath
from models.blocks.EfficientBlock import EfficientBlock

# 在文件开头添加
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
# from .blocks.ACmixMerge import ACmix
# from .blocks.Mambaout import GatedCNNBlock

import math

# 250410 ACmix中间两层融合思路验证

class BasicConv2d(nn.Module):
    """
    定义一个二维卷积层，包含卷积、批量归一化和 ReLU 激活函数。

    参数:
    in_planes (int): 输入特征图的通道数。
    out_planes (int): 输出特征图的通道数。
    kernel_size (int or tuple): 卷积核的大小。
    stride (int or tuple, 可选): 卷积的步长，默认为 1。
    padding (int or tuple, 可选): 填充的大小，默认为 0。
    dilation (int or tuple, 可选): 卷积核元素之间的间距，默认为 1。
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        # 定义卷积层，不使用偏置项
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # 定义批量归一化层，对输出特征图的每个通道进行归一化处理
        self.bn = nn.BatchNorm2d(out_planes)
        # 定义 ReLU 激活函数，使用 inplace=True 可以节省内存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播函数，定义模块的计算流程。

        参数:
        x (torch.Tensor): 输入的特征图。

        返回:
        torch.Tensor: 经过卷积和批量归一化处理后的特征图。
        """
        # 对输入特征图进行卷积操作
        x = self.conv(x)
        # 对卷积结果进行批量归一化处理
        x = self.bn(x)

        return x

class BasicConv2dReLu(nn.Module):
    """
    定义一个二维卷积层，包含卷积、批量归一化和 ReLU 激活函数。

    参数:
    in_planes (int): 输入特征图的通道数。
    out_planes (int): 输出特征图的通道数。
    kernel_size (int or tuple): 卷积核的大小。
    stride (int or tuple, 可选): 卷积的步长，默认为 1。
    padding (int or tuple, 可选): 填充的大小，默认为 0。
    dilation (int or tuple, 可选): 卷积核元素之间的间距，默认为 1。
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        # 调用父类 nn.Module 的构造函数
        super(BasicConv2dReLu, self).__init__()
        # 定义卷积层，输入通道数为 in_planes，输出通道数为 out_planes
        # 卷积核大小由 kernel_size 指定，步长为 stride，填充大小为 padding，膨胀率为 dilation
        # 不使用偏置项，因为后续会使用批量归一化层
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        # 定义批量归一化层，对卷积层的输出进行归一化处理
        # 可以加速模型收敛，提高模型的稳定性
        self.bn = nn.BatchNorm2d(out_planes)
        # 定义 ReLU 激活函数，引入非线性因素，增强模型的表达能力
        # inplace=True 表示直接在原张量上进行操作，节省内存
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        前向传播函数，定义模块的计算流程。

        参数:
        x (torch.Tensor): 输入的特征图。

        返回:
        torch.Tensor: 经过卷积、批量归一化和 ReLU 激活处理后的特征图。
        """
        # 对输入特征图进行卷积操作
        x = self.conv(x)
        # 对卷积结果进行批量归一化处理
        x = self.bn(x)
        # 对归一化后的结果应用 ReLU 激活函数
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 修改为3层解码器
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(32, 32, 3, padding=1)
        )

        # 直接从x4和x2开始处理，跳过原来的x3层
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # BasicConv2d(64, 32, 3, padding=1)  # 输入变为64因为只拼接x4和x2
        )

        self.decoder1 = nn.Sequential(
            BasicConv2d(64, 32, 3, padding=1)  # 输入保持64因为拼接x2和x1
        )

        self.EfficientBlock = EfficientBlock(64,32, 3, 3, 4, 1)
        
        self.conv = nn.Conv2d(channel, 1, 1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

    def forward(self, x4, x2x3, x1):
        # 处理x4 (32x11x11 -> 32x22x22)
        x4_decoder = self.decoder4(x4)  # 32x22x22

        # 直接连接x4和x2 (32x22x22 + 32x44x44)
        # 需要先对x4_decoder进行上采样
        x4_up = F.interpolate(x4_decoder, scale_factor=2, mode='bilinear', align_corners=True)  # 32x44x44

        x2x3x4_cat = torch.cat([x4_up, x2x3], dim=1)  # 64x44x44
        #
        x2x3x4_cat=self.EfficientBlock(x2x3x4_cat) #32x44x44
        
        x2x3x4_decoder = self.decoder2(x2x3x4_cat)  # 32x88x88

        # 连接x2和x1
        x_all_cat = torch.cat([x2x3x4_decoder, x1], 1)  # 64x88x88
        # x1_decoder = self.decoder1(x1_cat)  # 32x88x88
        x_all_cat=self.EfficientBlock(x_all_cat) #32x88x88
        x = self.conv(x_all_cat)  # 1x88x88
        x = self.upsample_4(x)  # 1x352x352
        return x

class Decode_mid_layer(nn.Module):
    def __init__(self, channel):
        super(Decode_mid_layer, self).__init__()

        # 添加用于深度监督的解码层
        self.decode_x1nor = nn.Sequential(
            # BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 1, 1),
            nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        )
        
        self.decode_x23nor = nn.Sequential(
            # BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 1, 1),
            nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)
        )
        
        self.decode_x4nor = nn.Sequential(
            # BasicConv2d(32, 32, 3, padding=1),
            nn.Conv2d(32, 1, 1),
            nn.Upsample(scale_factor=32, mode='bilinear', align_corners=True)
        )
    def forward(self, x1, x23, x4):
        x1 = self.decode_x1nor(x1)
        x23 = self.decode_x23nor(x23)
        x4 = self.decode_x4nor(x4)
        return x1, x23, x4

class DBANet(nn.Module):
    """
    DBANet 模型类，继承自 nn.Module。
    结合骨干网络（pvt_v2_b2）、通道归一化层、ACmix 模块、GatedCNNBlock 和解码器，最后通过 Sigmoid 激活函数输出预测结果。
    """
    def __init__(self, channel=32):
        """
        初始化 DBANet 模型。

        参数:
        channel (int): 通道归一化层的输出通道数，默认为 32。
        """
        super(DBANet, self).__init__()

        # 加载骨干网络 pvt_v2_b2，其输出通道数分别为 [64, 128, 320, 512]
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        # 预训练模型的路径
        abspath = os.path.abspath(__file__) # 获取所执行脚本的绝对路径
        proj_path = os.path.dirname(abspath) # 获取父级路径
        path = proj_path + '/backbone/pretrained/pvt_v2_b2.pth'
        
        # 加载预训练模型的权重
        save_model = torch.load(path)
        # 获取当前骨干网络的状态字典
        model_dict = self.backbone.state_dict()
        # 筛选出预训练模型中与当前骨干网络匹配的权重
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        # 更新当前骨干网络的状态字典
        model_dict.update(state_dict)
        # 加载更新后的状态字典到骨干网络
        self.backbone.load_state_dict(model_dict)

        # 输入尺寸为 3x352x352
        # 通道归一化层，将输入通道数转换为指定的通道数
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 1, 1) # 128x44x44->32x44x44
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        self.tiaozheng = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            BasicConv2d(320, 128, kernel_size=1, padding=0)  # 使用1x1卷积更高效
        )

        self.AcMerge=ACmix(128,128)


        # # ACmix 模块，用于特征增强
        # self.ACmix1 = ACmix(64,64)
        # self.ACmix2 = ACmix(128,128)
        # self.ACmix3 = ACmix(320,320)
        # self.ACmix4 = ACmix(512,512)  

        # 新增 GatedCNNBlock 实例，对应骨干网络不同阶段输出
        self.gated_cnn_block1 = GatedCNNBlock(dim=64)
        self.gated_cnn_block2 = GatedCNNBlock(dim=128)
        self.gated_cnn_block3 = GatedCNNBlock(dim=320)
        self.gated_cnn_block4 = GatedCNNBlock(dim=512)

        # 解码器，用于将特征图解码为预测结果
        self.Decoder = Decoder(channel)

        #中间层解码器,用于解码中间层特征图
        self.Decode_mid_layer = Decode_mid_layer(channel=32)

        # Sigmoid 激活函数，用于将输出转换为概率值
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播函数，定义模型的计算流程。

        参数:
        x (torch.Tensor): 输入的张量，尺寸为 [batch_size, 3, 352, 352]

        返回:
        tuple: 包含原始预测结果和经过 Sigmoid 激活后的预测结果
        """
        # 通过骨干网络进行特征提取
        pvt = self.backbone(x)
        # 提取不同阶段的特征图
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        # 调整特征图维度以适应 GatedCNNBlock 输入要求 [B, C, H, W] -> [B, H, W, C]
        x1_permuted = x1.permute(0, 2, 3, 1)
        x2_permuted = x2.permute(0, 2, 3, 1)
        x3_permuted = x3.permute(0, 2, 3, 1)
        x4_permuted = x4.permute(0, 2, 3, 1)

        # 通过 GatedCNNBlock 处理特征
        x1_gated = self.gated_cnn_block1(x1_permuted)
        x2_gated = self.gated_cnn_block2(x2_permuted)
        x3_gated = self.gated_cnn_block3(x3_permuted)
        x4_gated = self.gated_cnn_block4(x4_permuted)

        # 调整回原始维度 [B, H, W, C] -> [B, C, H, W]
        x1_gated = x1_gated.permute(0, 3, 1, 2)
        x2_gated = x2_gated.permute(0, 3, 1, 2)
        x3_gated = x3_gated.permute(0, 3, 1, 2)
        x4_gated = x4_gated.permute(0, 3, 1, 2)

        #====将23层使用ACmix2融合==================

        x3_gated=self.tiaozheng(x3_gated)  # 16, 128, 44, 44
        x2x3= self.AcMerge(x2_gated,x3_gated) # [16, 128, 44, 44]
        # print(x2x3.shape)
        # exit()


        # 通过通道归一化层将特征图的通道数统一
        x1_nor = self.ChannelNormalization_1(x1_gated) # 32x88x88
        x2x3_nor = self.ChannelNormalization_2(x2x3) # 32x44x44
        # x3_nor = self.ChannelNormalization_3(x3_all) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4_gated) # 32x11x11

        # 通过解码器进行解码，得到预测结果
        prediction = self.Decoder(x4_nor, x2x3_nor, x1_nor) # 1, 352, 352

        # 分别获取中间层的解码结果,用于深度监督
        x1_dec, x23_dec, x4_dec = self.Decode_mid_layer(x1_nor, x2x3_nor, x4_nor) # 1, 352, 352
        # 返回原始预测结果和经过 Sigmoid 激活后的预测结果
        return prediction, self.sigmoid(prediction),x1_dec,self.sigmoid(x1_dec),x23_dec,self.sigmoid(x23_dec),x4_dec,self.sigmoid(x4_dec)  # 1, 352, 352
if __name__ == '__main__':
    # 实例化 DBANet 模型
    model = DBANet()
    # 打印模型的结构
    print(model)
    # 创建一个随机输入张量，尺寸为 [batch_size, 3, 352, 352]
    x = torch.randn(16, 3, 352, 352)
    model(x)
