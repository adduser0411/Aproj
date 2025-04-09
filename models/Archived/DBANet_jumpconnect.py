import torch
import torch.nn as nn
import torch.nn.functional as F
from ..backbone.pvtv2 import pvt_v2_b2
import torch.nn.functional as F
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax, Dropout
from torch.jit import Final

from typing import List, Callable
from torch import Tensor

import math


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)

        return x
        
class BasicConv2dReLu(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2dReLu, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channel):
        super(Decoder, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 32*11*11
        self.decoder4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(32, 32, 3, padding=1)  
        )
               
        # 32*22*22
        self.decoder3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(64, 32, 3, padding=1) 
        )
        
        # 32*44*44
        self.decoder2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),  
            BasicConv2d(64, 32, 3, padding=1)  
        )
        
        # 32*88*88
        self.decoder1 = nn.Sequential(
            BasicConv2d(64, 32, 3, padding=1)  
        )
        
        self.conv = nn.Conv2d(channel, 1, 1)
        self.upsample_4 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)       
        
                # 新增以下四行，添加卷积层用于调整通道数
        self.adjust_channel_4 = nn.Conv2d(512, 32, kernel_size=1)  # 修改行 1
        self.adjust_channel_3 = nn.Conv2d(320, 32, kernel_size=1)  # 修改行 2
        self.adjust_channel_2 = nn.Conv2d(128, 32, kernel_size=1)  # 修改行 3
        self.adjust_channel_1 = nn.Conv2d(64, 32, kernel_size=1)  # 修改行 4
         
        
    def forward(self, x4, x3, x2, x1, encoder_x4, encoder_x3, encoder_x2, encoder_x1):
        # 添加跳跃链接
        
# - 首先，将输入的 `x4` 通过 `self.decoder4` 模块进行处理，得到 `x4_decoder`。
# - 然后，将 `x4_decoder` 与编码器输出的 `encoder_x4` 相加，实现跳跃连接。这样做的目的是将编码器中该层的特征信息直接传递到解码器的对应层，使得解码器能够利用这些特征。
        x4_decoder = self.decoder4(x4)  # 32*22*22
        # 修改行 1: 对 encoder_x4 进行上采样
        encoder_x4_up = self.upsample(encoder_x4)
        # 新增以下一行，调整 encoder_x4_up 的通道数
        encoder_x4_up = self.adjust_channel_4(encoder_x4_up)  # 修改行 5
        x4_decoder = x4_decoder + encoder_x4_up  # 跳跃链接
# - 先将 `x4_decoder` 和 `x3` 在通道维度上进行拼接，得到 `x3_cat`。
# - 接着将 `x3_cat` 通过 `self.decoder3` 模块进行处理，得到 `x3_decoder`。
# - 最后将 `x3_decoder` 与编码器输出的 `encoder_x3` 相加，实现跳跃连接。
        x3_cat = torch.cat([x4_decoder, x3], 1)   # 64*22*22       
        x3_decoder = self.decoder3(x3_cat)
        # 修改行 2: 对 encoder_x3 进行上采样
        encoder_x3_up = self.upsample(encoder_x3)
        # 新增以下一行，调整 encoder_x3_up 的通道数
        encoder_x3_up = self.adjust_channel_3(encoder_x3_up)  # 修改行 6
        x3_decoder = x3_decoder + encoder_x3_up  # 跳跃链接
# - 把 `x3_decoder` 和 `x2` 在通道维度上拼接，得到 `x2_cat`。
# - 将 `x2_cat` 通过 `self.decoder2` 模块处理，得到 `x2_decoder`。
# - 把 `x2_decoder` 与编码器输出的 `encoder_x2` 相加，实现跳跃连接。        
        x2_cat = torch.cat([x3_decoder, x2], 1) # 64*44*44        
        x2_decoder = self.decoder2(x2_cat)   # 32*88*88        
        # 修改行 3: 对 encoder_x2 进行上采样
        encoder_x2_up = self.upsample(encoder_x2)
        # 新增以下一行，调整 encoder_x2_up 的通道数
        encoder_x2_up = self.adjust_channel_2(encoder_x2_up)  # 修改行 7
        x2_decoder = x2_decoder + encoder_x2_up  # 跳跃链接
# - 把 `x2_decoder` 和 `x1` 在通道维度上拼接，得到 `x1_cat`。
# - 将 `x1_cat` 通过 `self.decoder1` 模块处理，得到 `x1_decoder`。
# - 把 `x1_decoder` 与编码器输出的 `encoder_x1` 相加，实现跳跃连接。        
        x1_cat = torch.cat([x2_decoder, x1], 1) # 64*88*88      
        x1_decoder = self.decoder1(x1_cat)   # 32*88*88
        # 新增以下一行，对 encoder_x1 进行通道调整
        encoder_x1 = self.adjust_channel_1(encoder_x1)  # 修改行 8
        x1_decoder = x1_decoder + encoder_x1  # 跳跃链接
        
        x = self.conv(x1_decoder) # 1*88*88
        x = self.upsample_4(x) # 1*352*352
        return x
  
# https://github.com/LeapLabTHU/ACmix
# https://arxiv.org/pdf/2111.14556

 # 定义一个函数来生成位置编码，返回一个包含位置信息的张量
def position(H, W, is_cuda=True):
    # 生成宽度和高度的位置信息，范围在-1到1之间
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)# 为宽度生成线性间距的位置信息并复制到GPU
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W) # 为高度生成线性间距的位置信息并复制到GPU
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1) # 在CPU上为宽度生成线性间距的位置信息
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W) # 在CPU上为高度生成线性间距的位置信息
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0) # 合并宽度和高度的位置信息，并增加一个维度
    return loc

# 定义一个函数实现步长操作，用于降采样
def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride] # 通过步长来降低采样率

# 初始化函数，将张量的值填充为0.5
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5) # 使用0.5来填充张量

# 初始化函数，将张量的值填充为0
def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)

# 定义ACmix模块的类
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__() # 调用父类的构造函数
        # 初始化模块参数
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))  # 注意力分支权重
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))  # 卷积分支权重
        self.head_dim = self.out_planes // self.head  # 每个头的维度

        # 定义用于特征变换的卷积层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)  # 位置编码的卷积层

        # 定义自注意力所需的padding和展开操作
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        # 定义用于生成动态卷积核的全连接层和深度可分离卷积层
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, out_planes,
                                  kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1,
                                  stride=stride)# 深度可分离卷积层，用于应用动态卷积核

        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        init_rate_half(self.rate1)  # 初始化注意力分支权重为0.5
        init_rate_half(self.rate2)  # 初始化卷积分支权重为0.5
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)# 设置为可学习参数
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)# 初始化偏置为0

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)# 应用转换层
        scaling = float(self.head_dim) ** -0.5# 缩放因子，用于自注意力计算
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride # 计算输出的高度和宽度

        pe = self.conv_p(position(h, w, x.is_cuda))# 生成位置编码
        # 为自注意力机制准备q, k, v
        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1: # 如果步长大于1，则对q和位置编码进行降采样
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe
       # 展开k和位置编码，准备自注意力计算
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim,
                                                         self.kernel_att * self.kernel_att, h_out,
                                                         w_out)  # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att * self.kernel_att, h_out,
                                                        w_out)  # 1, head_dim, k_att^2, h_out, w_out
		 # 计算注意力权重
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(
            1)  # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)
		  # 应用注意力权重
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att * self.kernel_att,
                                                        h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)
		# 动态卷积核
        f_all = self.fc(torch.cat(
            [q.view(b, self.head, self.head_dim, h * w), k.view(b, self.head, self.head_dim, h * w),
             v.view(b, self.head, self.head_dim, h * w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])

        out_conv = self.dep_conv(f_conv)
		# 将注意力分支和卷积分支的输出相加
        return self.rate1 * out_att + self.rate2 * out_conv


class DBANet(nn.Module):
    def __init__(self, channel=32):
        super(DBANet, self).__init__()

        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        path = './models/backbone/pretrained/pvt_v2_b2.pth'
        save_model = torch.load(path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)

        # input 3x352x352
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 1, 1) # 128x44x44->32x44x44
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1) # 320x22x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1) # 512x11x11->32x11x11

        self.ACmix1 = ACmix(64,64)
        self.ACmix2 = ACmix(128,128)
        self.ACmix3 = ACmix(320,320)
        self.ACmix4 = ACmix(512,512)  
        
        self.Decoder = Decoder(channel)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # backbone
        pvt = self.backbone(x)
        x1 = pvt[0] # 64x88x88
        x2 = pvt[1] # 128x44x44
        x3 = pvt[2] # 320x22x22
        x4 = pvt[3] # 512x11x11

        x1_dea = self.ACmix1(x1)
        x2_dea = self.ACmix2(x2)
        x3_dea = self.ACmix3(x3)
        x4_dea = self.ACmix4(x4)

        x1_all =  x1_dea
        x2_all =  x2_dea
        x3_all =  x3_dea
        x4_all =  x4_dea

# 这里将编码器输出的 x4、x3、x2、x1 作为额外的参数传递给 Decoder 类的 forward 方法，以便在解码器中实现跳跃连接。        
        x1_nor = self.ChannelNormalization_1(x1_all) # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2_all) # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3_all) # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4_all) # 32x11x11
        
        # 将编码器的特征传递给解码器
        prediction = self.Decoder(x4_nor, x3_nor, x2_nor, x1_nor, x4, x3, x2, x1)

        return prediction, self.sigmoid(prediction)
