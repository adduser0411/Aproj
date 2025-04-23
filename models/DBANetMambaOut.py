import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.pvtv2 import pvt_v2_b2
import os
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
from .blocks.ACmix import ACmix
from .blocks.Mambaout import GatedCNNBlock

import math


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
        """
        初始化 Decoder 类。

        参数:
        channel (int): 通道数，用于定义卷积层的输入和输出通道数。
        """
        # 调用父类 nn.Module 的构造函数
        super(Decoder, self).__init__()

        # 定义一个通用的上采样层，用于将特征图的尺寸放大两倍
        # scale_factor=2 表示放大倍数为 2
        # mode='bilinear' 表示使用双线性插值进行上采样
        # align_corners=True 表示对齐角点像素
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # 解码器第4层，输入为 32x11x11 的特征图
        # 首先将特征图尺寸放大两倍，然后通过一个基本卷积层进行特征提取
        self.decoder4 = nn.Sequential(
            # 上采样层，将特征图尺寸从 32x11x11 放大到 32x22x22
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 基本卷积层，输入通道数为 32，输出通道数为 32，卷积核大小为 3，填充为 1
            # 经过该卷积层后，特征图尺寸保持 32x22x22 不变
            BasicConv2d(32, 32, 3, padding=1)
        )

        # 解码器第3层，输入为 32x22x22 的特征图
        # 同样先将特征图尺寸放大两倍，再通过基本卷积层进行特征提取
        self.decoder3 = nn.Sequential(
            # 上采样层，将特征图尺寸从 32x22x22 放大到 32x44x44
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 基本卷积层，输入通道数为 64（因为会和上一层的输出拼接），输出通道数为 32，卷积核大小为 3，填充为 1
            # 经过该卷积层后，特征图尺寸为 32x44x44
            BasicConv2d(64, 32, 3, padding=1)
        )

        # 解码器第2层，输入为 32x44x44 的特征图
        # 操作与前面两层类似
        self.decoder2 = nn.Sequential(
            # 上采样层，将特征图尺寸从 32x44x44 放大到 32x88x88
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            # 基本卷积层，输入通道数为 64（因为会和上一层的输出拼接），输出通道数为 32，卷积核大小为 3，填充为 1
            # 经过该卷积层后，特征图尺寸为 32x88x88
            BasicConv2d(64, 32, 3, padding=1)
        )

        # 解码器第1层，输入为 32x88x88 的特征图
        # 只通过一个基本卷积层进行特征提取
        self.decoder1 = nn.Sequential(
            # 基本卷积层，输入通道数为 64（因为会和上一层的输出拼接），输出通道数为 32，卷积核大小为 3，填充为 1
            # 经过该卷积层后，特征图尺寸保持 32x88x88 不变
            BasicConv2d(64, 32, 3, padding=1)
        )

        # 定义一个卷积层，将 32 通道的特征图转换为 1 通道的预测结果
        # 输入通道数为 channel（即 32），输出通道数为 1，卷积核大小为 1
        # 经过该卷积层后，特征图尺寸为 1x88x88
        self.conv = nn.Conv2d(channel, 1, 1)

        # 定义一个上采样层，将 1x88x88 的特征图放大 4 倍，得到 1x352x352 的最终预测结果
        # scale_factor=4 表示放大倍数为 4
        # mode='bilinear' 表示使用双线性插值进行上采样
        # align_corners=True 表示对齐角点像素
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)


    def forward(self, x4, x3, x2, x1):
        """
        前向传播函数，定义解码器的计算流程。

        参数:
        x4 (torch.Tensor): 第4层的特征图，尺寸为 32x11x11
        x3 (torch.Tensor): 第3层的特征图，尺寸为 32x22x22
        x2 (torch.Tensor): 第2层的特征图，尺寸为 32x44x44
        x1 (torch.Tensor): 第1层的特征图，尺寸为 32x88x88

        返回:
        torch.Tensor: 最终的预测结果，尺寸为 1x352x352
        """
        # 通过解码器第4层处理第4层的特征图
        # 第4层特征图 x4 尺寸为 32x11x11，经过解码器第4层后尺寸变为 32x22x22
        x4_decoder = self.decoder4(x4)  # 32*22*22

        # 将第4层解码器的输出和第3层的特征图在通道维度（dim=1）上拼接
        # x4_decoder 尺寸为 32x22x22，x3 尺寸为 32x22x22，拼接后得到 64x22x22 的特征图
        x3_cat = torch.cat([x4_decoder, x3], 1)   # 64*22*22
        # 通过解码器第3层处理拼接后的特征图
        # 拼接后的特征图 x3_cat 尺寸为 64x22x22，经过解码器第3层后尺寸变为 32x44x44
        x3_decoder = self.decoder3(x3_cat)

        # 将第3层解码器的输出和第2层的特征图在通道维度（dim=1）上拼接
        # x3_decoder 尺寸为 32x44x44，x2 尺寸为 32x44x44，拼接后得到 64x44x44 的特征图
        x2_cat = torch.cat([x3_decoder, x2], 1) # 64*44*44
        # 通过解码器第2层处理拼接后的特征图
        # 拼接后的特征图 x2_cat 尺寸为 64x44x44，经过解码器第2层后尺寸变为 32x88x88
        x2_decoder = self.decoder2(x2_cat)   # 32*88*88

        # 将第2层解码器的输出和第1层的特征图在通道维度（dim=1）上拼接
        # x2_decoder 尺寸为 32x88x88，x1 尺寸为 32x88x88，拼接后得到 64x88x88 的特征图
        x1_cat = torch.cat([x2_decoder, x1], 1) # 64*88*88
        # 通过解码器第1层处理拼接后的特征图
        # 拼接后的特征图 x1_cat 尺寸为 64x88x88，经过解码器第1层后尺寸变为 32x88x88
        x1_decoder = self.decoder1(x1_cat)   # 32*88*88

        # 通过卷积层将 32 通道的特征图转换为 1 通道的预测结果
        # x1_decoder 尺寸为 32x88x88，经过卷积层后尺寸变为 1x88x88
        x = self.conv(x1_decoder) # 1*88*88
        # 通过上采样层将 1x88x88 的特征图放大 4 倍，得到最终的预测结果
        # 上采样后特征图 x 的尺寸变为 1x352x352
        x = self.upsample_4(x) # 1*352*352

        return x

 # 定义一个函数来生成位置编码，返回一个包含位置信息的张量



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

        # ACmix 模块，用于特征增强
        self.ACmix1 = ACmix(64,64)
        self.ACmix2 = ACmix(128,128)
        self.ACmix3 = ACmix(320,320)
        self.ACmix4 = ACmix(512,512)  

        # 新增 GatedCNNBlock 实例，对应骨干网络不同阶段输出
        self.gated_cnn_block1 = GatedCNNBlock(dim=64)
        self.gated_cnn_block2 = GatedCNNBlock(dim=128)
        self.gated_cnn_block3 = GatedCNNBlock(dim=320)
        self.gated_cnn_block4 = GatedCNNBlock(dim=512)

        # 解码器，用于将特征图解码为预测结果
        self.Decoder = Decoder(channel)

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

        # 通过 ACmix 模块对特征图进行增强
        x1_dea = self.ACmix1(x1)
        x2_dea = self.ACmix2(x2)
        x3_dea = self.ACmix3(x3)
        x4_dea = self.ACmix4(x4)

        # 调整特征图维度以适应 GatedCNNBlock 输入要求 [B, C, H, W] -> [B, H, W, C]
        x1_permuted = x1_dea.permute(0, 2, 3, 1)
        x2_permuted = x2_dea.permute(0, 2, 3, 1)
        x3_permuted = x3_dea.permute(0, 2, 3, 1)
        x4_permuted = x4_dea.permute(0, 2, 3, 1)

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

        # 这里可以选择将 GatedCNNBlock 输出替代或融合到原有流程，此处选择融合
        x1_all = x1_dea + x1_gated
        x2_all = x2_dea + x2_gated
        x3_all = x3_dea + x3_gated
        x4_all = x4_dea + x4_gated

        # 通过通道归一化层将特征图的通道数统一
        x1_nor = self.ChannelNormalization_1(x1_all) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2_all) # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3_all) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4_all) # 32x11x11

        # 通过解码器进行解码，得到预测结果
        prediction = self.Decoder(x4_nor, x3_nor, x2_nor, x1_nor)

        # 返回原始预测结果和经过 Sigmoid 激活后的预测结果
        return prediction, self.sigmoid(prediction)

if __name__ == '__main__':
    # 实例化 DBANet 模型
    model = DBANet()
    # 打印模型的结构