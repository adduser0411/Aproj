
import torch
import torch.nn as nn
from typing import List
from models.blocks.ShuffleAttention import ShuffleAttention
# from .GCSA import GCSA
class ConvFFN(nn.Module):

    def __init__(self, in_channels, hidden_channels, kernel_size, stride,
                 out_channels, act_layer=nn.GELU, drop_out=0.):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1, 1, 0)
        self.act = act_layer()
        self.dwconv = nn.Conv2d(hidden_channels, hidden_channels, kernel_size, stride, 
                                kernel_size//2, groups=hidden_channels)
        self.fc2 = nn.Conv2d(hidden_channels, out_channels, 1, 1, 0)
        self.drop = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor):
        '''
        x: (b h w c)
        '''
        x = self.fc1(x)
        x = self.act(x)
        x = self.dwconv(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()

    def forward(self, x):
        return x
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

class EfficientBlock(nn.Module):
    def __init__(self, channel,outchannel, kernel_sizes: List[int], mlp_kernel_size: int, mlp_ratio: int, stride: int, mlp_drop=0., drop_path=0.):
        super().__init__()
        self.channel = channel
        self.mlp_ratio = mlp_ratio
        self.norm1 = nn.GroupNorm(1, channel)
        # self.attn = GCSA(channel)
        self.drop_path = DropPath(drop_path)
        self.norm2 = nn.GroupNorm(1, channel)
        mlp_hidden_dim = int(channel * mlp_ratio)
        
        self.mlp = ConvFFN(channel, mlp_hidden_dim, mlp_kernel_size, stride, channel, drop_out=mlp_drop)
        # self.conv = BasicConv2d(channel, outchannel, 3, padding=1)
        self.conv = BasicConv2d(channel, channel, 3, padding=1)   #0421师兄调整
        self.ShuffleAttention=ShuffleAttention(channel,4)
        
    def forward(self, x: torch.Tensor):
        x1 = x
        x2 = self.drop_path(self.ShuffleAttention(self.norm1(x)))
        x3 = self.drop_path(self.mlp(self.norm2(x)))
        
        x_all = x1 + x2 + x3
        # x_all = self.conv(x_all)   
        x_all = self.conv(x_all)+x  #0421师兄调整

        return x_all