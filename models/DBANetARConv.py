import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone.pvtv2 import pvt_v2_b2
import os
from torch.nn import Softmax, Dropout
from torch.jit import Final
from typing import List, Callable
from torch import Tensor
import math


# Adaptive Rectangular Convolution (ARConv).
class ARConv(nn.Module):
    def __init__(self, inc, outc, kernel_size=3, padding=1, stride=1, l_max=9, w_max=9, flag=False, modulation=True):
        super(ARConv, self).__init__()
        self.lmax = l_max
        self.wmax = w_max
        self.inc = inc
        self.outc = outc
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride
        self.zero_padding = nn.ZeroPad2d(padding)
        self.flag = flag
        self.modulation = modulation
        self.i_list = [33, 35, 53, 37, 73, 55, 57, 75, 77]
        self.convs = nn.ModuleList(
            [
                nn.Conv2d(inc, outc, kernel_size=(i // 10, i % 10), stride=(i // 10, i % 10), padding=0)
                for i in self.i_list
            ]
        )
        self.m_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.Tanh()
        )
        self.b_conv = nn.Sequential(
            nn.Conv2d(inc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Conv2d(outc, outc, kernel_size=3, padding=1, stride=stride)
        )
        self.p_conv = nn.Sequential(
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(inc, inc, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(inc),
            nn.LeakyReLU(),
        )
        self.l_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.w_conv = nn.Sequential(
            nn.Conv2d(inc, 1, kernel_size=3, padding=1, stride=stride),
            nn.BatchNorm2d(1),
            nn.LeakyReLU(),
            nn.Dropout2d(0),
            nn.Conv2d(1, 1, 1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.dropout1 = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.3)
        self.hook_handles = []
        self.hook_handles.append(self.m_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.m_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.b_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.p_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.l_conv[1].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[0].register_full_backward_hook(self._set_lr))
        self.hook_handles.append(self.w_conv[1].register_full_backward_hook(self._set_lr))

        self.reserved_NXY = nn.Parameter(torch.tensor([3, 3], dtype=torch.int32), requires_grad=False)

    @staticmethod
    def _set_lr(module, grad_input, grad_output):
        grad_input = tuple(g * 0.1 if g is not None else None for g in grad_input)
        grad_output = tuple(g * 0.1 if g is not None else None for g in grad_output)
        return grad_input

    def remove_hooks(self):
        for handle in self.hook_handles:
            handle.remove()  # 移除钩子函数
        self.hook_handles.clear()  # 清空句柄列表

    def forward(self, x, epoch, hw_range):
        assert isinstance(hw_range, list) and len(hw_range) == 2, "hw_range should be a list with 2 elements, represent the range of h w"
        scale = hw_range[1] // 9
        if hw_range[0] == 1 and hw_range[1] == 3:
            scale = 1
        m = self.m_conv(x)
        bias = self.b_conv(x)
        offset = self.p_conv(x * 100)
        l = self.l_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        w = self.w_conv(offset) * (hw_range[1] - 1) + 1  # b, 1, h, w
        if epoch <= 100:
            mean_l = l.mean(dim=0).mean(dim=1).mean(dim=1)
            mean_w = w.mean(dim=0).mean(dim=1).mean(dim=1)
            N_X = int(mean_l // scale)
            N_Y = int(mean_w // scale)
            def phi(x):
                if x % 2 == 0:
                    x -= 1
                return x
            N_X, N_Y = phi(N_X), phi(N_Y)
            N_X, N_Y = max(N_X, 3), max(N_Y, 3)
            N_X, N_Y = min(N_X, 7), min(N_Y, 7)
            if epoch == 100:
                self.reserved_NXY = nn.Parameter(
                    torch.tensor([N_X, N_Y], dtype=torch.int32, device=x.device),
                    requires_grad=False
                )
        else:
            N_X = self.reserved_NXY[0]
            N_Y = self.reserved_NXY[1]
        N = N_X * N_Y
        # print(N_X, N_Y)
        l = l.repeat([1, N, 1, 1])
        w = w.repeat([1, N, 1, 1])
        offset = torch.cat((l, w), dim=1)
        dtype = offset.data.type()
        if self.padding:
            x = self.zero_padding(x)
        p = self._get_p(offset, dtype, N_X, N_Y)  # (b, 2*N, h, w)
        p = p.contiguous().permute(0, 2, 3, 1)  # (b, h, w, 2*N)
        q_lt = p.detach().floor()
        q_rb = q_lt + 1
        q_lt = torch.cat(
            [
                torch.clamp(q_lt[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_lt[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_rb = torch.cat(
            [
                torch.clamp(q_rb[..., :N], 0, x.size(2) - 1),
                torch.clamp(q_rb[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        ).long()
        q_lb = torch.cat([q_lt[..., :N], q_rb[..., N:]], dim=-1)
        q_rt = torch.cat([q_rb[..., :N], q_lt[..., N:]], dim=-1)
        # clip p
        p = torch.cat(
            [
                torch.clamp(p[..., :N], 0, x.size(2) - 1),
                torch.clamp(p[..., N:], 0, x.size(3) - 1),
            ],
            dim=-1,
        )
        # bilinear kernel (b, h, w, N)
        g_lt = (1 + (q_lt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_lt[..., N:].type_as(p) - p[..., N:])
        )
        g_rb = (1 - (q_rb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_rb[..., N:].type_as(p) - p[..., N:])
        )
        g_lb = (1 + (q_lb[..., :N].type_as(p) - p[..., :N])) * (
                1 - (q_lb[..., N:].type_as(p) - p[..., N:])
        )
        g_rt = (1 - (q_rt[..., :N].type_as(p) - p[..., :N])) * (
                1 + (q_rt[..., N:].type_as(p) - p[..., N:])
        )
        # (b, c, h, w, N)
        x_q_lt = self._get_x_q(x, q_lt, N)
        x_q_rb = self._get_x_q(x, q_rb, N)
        x_q_lb = self._get_x_q(x, q_lb, N)
        x_q_rt = self._get_x_q(x, q_rt, N)
        # (b, c, h, w, N)
        x_offset = (
                g_lt.unsqueeze(dim=1) * x_q_lt
                + g_rb.unsqueeze(dim=1) * x_q_rb
                + g_lb.unsqueeze(dim=1) * x_q_lb
                + g_rt.unsqueeze(dim=1) * x_q_rt
        )
        x_offset = self._reshape_x_offset(x_offset, N_X, N_Y)
        x_offset = self.dropout2(x_offset)
        x_offset = self.convs[self.i_list.index(N_X * 10 + N_Y)](x_offset)
        out = x_offset * m + bias
        return out

    def _get_p_n(self, N, dtype, n_x, n_y):
        p_n_x, p_n_y = torch.meshgrid(
            torch.arange(-(n_x - 1) // 2, (n_x - 1) // 2 + 1),
            torch.arange(-(n_y - 1) // 2, (n_y - 1) // 2 + 1),
            indexing = 'ij',
        )
        p_n = torch.cat([torch.flatten(p_n_x), torch.flatten(p_n_y)], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_0(self, h, w, N, dtype):
        p_0_x, p_0_y = torch.meshgrid(
            torch.arange(1, h * self.stride + 1, self.stride),
            torch.arange(1, w * self.stride + 1, self.stride),
            indexing = 'ij',
        )
        p_0_x = torch.flatten(p_0_x).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0_y = torch.flatten(p_0_y).view(1, 1, h, w).repeat(1, N, 1, 1)
        p_0 = torch.cat([p_0_x, p_0_y], 1).type(dtype)
        return p_0

    def _get_p(self, offset, dtype, n_x, n_y):
        N, h, w = offset.size(1) // 2, offset.size(2), offset.size(3)
        L, W = offset.split([N, N], dim=1)
        L = L / n_x
        W = W / n_y
        offsett = torch.cat([L, W], dim=1)
        p_n = self._get_p_n(N, dtype, n_x, n_y)
        p_n = p_n.repeat([1, 1, h, w])
        p_0 = self._get_p_0(h, w, N, dtype)
        p = p_0 + offsett * p_n
        return p

    def _get_x_q(self, x, q, N):
        b, h, w, _ = q.size()
        padded_w = x.size(3)
        c = x.size(1)
        x = x.contiguous().view(b, c, -1)
        index = q[..., :N] * padded_w + q[..., N:]
        index = (
            index.contiguous()
            .unsqueeze(dim=1)
            .expand(-1, c, -1, -1, -1)
            .contiguous()
            .view(b, c, -1)
        )
        x_offset = x.gather(dim=-1, index=index).contiguous().view(b, c, h, w, N)
        return x_offset

    @staticmethod
    def _reshape_x_offset(x_offset, n_x, n_y):
        b, c, h, w, N = x_offset.size()
        x_offset = torch.cat([x_offset[..., s:s + n_y].contiguous().view(b, c, h, w * n_y) for s in range(0, N, n_y)],
                             dim=-1)
        x_offset = x_offset.contiguous().view(b, c, h * n_x, w * n_y)
        return x_offset


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
        # 应用 ReLU 激活函数
        x = self.relu(x)
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
        x3_cat = torch.cat([x4_decoder, x3], 1)  # 64*22*22
        # 通过解码器第3层处理拼接后的特征图
        # 拼接后的特征图 x3_cat 尺寸为 64x22x22，经过解码器第3层后尺寸变为 32x44x44
        x3_decoder = self.decoder3(x3_cat)

        # 将第3层解码器的输出和第2层的特征图在通道维度（dim=1）上拼接
        # x3_decoder 尺寸为 32x44x44，x2 尺寸为 32x44x44，拼接后得到 64x44x44 的特征图
        x2_cat = torch.cat([x3_decoder, x2], 1)  # 64*44*44
        # 通过解码器第2层处理拼接后的特征图
        # 拼接后的特征图 x2_cat 尺寸为 64x44x44，经过解码器第2层后尺寸变为 32x88x88
        x2_decoder = self.decoder2(x2_cat)  # 32*88*88

        # 将第2层解码器的输出和第1层的特征图在通道维度（dim=1）上拼接
        # x2_decoder 尺寸为 32x88x88，x1 尺寸为 32x88x88，拼接后得到 64x88x88 的特征图
        x1_cat = torch.cat([x2_decoder, x1], 1)  # 64*88*88
        # 通过解码器第1层处理拼接后的特征图
        # 拼接后的特征图 x1_cat 尺寸为 64x88x88，经过解码器第1层后尺寸变为 32x88x88
        x1_decoder = self.decoder1(x1_cat)  # 32*88*88

        # 通过卷积层将 32 通道的特征图转换为 1 通道的预测结果
        # x1_decoder 尺寸为 32x88x88，经过卷积层后尺寸变为 1x88x88
        x = self.conv(x1_decoder)  # 1*88*88
        # 通过上采样层将 1x88x88 的特征图放大 4 倍，得到最终的预测结果
        # 上采样后特征图 x 的尺寸变为 1x352x352
        x = self.upsample_4(x)  # 1*352*352

        return x


# 定义一个函数来生成位置编码，返回一个包含位置信息的张量
def position(H, W, is_cuda=True):
    # 生成宽度和高度的位置信息，范围在-1到1之间
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)  # 为宽度生成线性间距的位置信息并复制到GPU
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)  # 为高度生成线性间距的位置信息并复制到GPU
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)  # 在CPU上为宽度生成线性间距的位置信息
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)  # 在CPU上为高度生成线性间距的位置信息
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)  # 合并宽度和高度的位置信息，并增加一个维度
    return loc


# 定义一个函数实现步长操作，用于降采样
def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]  # 通过步长来降低采样率


# 初始化函数，将张量的值填充为0.5
def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)  # 使用0.5来填充张量


# 初始化函数，将张量的值填充为0
def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


# 定义ACmix模块的类
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()  # 调用父类的构造函数
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
                                  stride=self.stride)  # 深度可分离卷积层，用于应用动态卷积核

        self.reset_parameters()  # 参数初始化

    def reset_parameters(self):
        init_rate_half(self.rate1)  # 初始化注意力分支权重为0.5
        init_rate_half(self.rate2)  # 初始化卷积分支权重为0.5
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i // self.kernel_conv, i % self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)  # 设置为可学习参数
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)  # 初始化偏置为0

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)  # 应用转换层
        scaling = float(self.head_dim) ** -0.5  # 缩放因子，用于自注意力计算
        b, c, h, w = q.shape
        h_out, w_out = h // self.stride, w // self.stride  # 计算输出的高度和宽度

        pe = self.conv_p(position(h, w, x.is_cuda))  # 生成位置编码
        # 为自注意力机制准备q, k, v
        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:  # 如果步长大于1，则对q和位置编码进行降采样
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
    """
    DBANet 模型类，继承自 nn.Module。
    结合骨干网络（pvt_v2_b2）、通道归一化层、ACmix 模块、ARConv 模块和解码器，最后通过 Sigmoid 激活函数输出预测结果。
    """
    def __init__(self, channel=32, epoch=1, hw_range=[1, 18]):
        """
        初始化 DBANet 模型。

        参数:
        channel (int): 通道归一化层的输出通道数，默认为 32。
        epoch (int): 当前训练轮次，用于 ARConv 模块。
        hw_range (list): 高度和宽度的学习范围，用于 ARConv 模块。
        """
        super(DBANet, self).__init__()
        self.epoch = epoch
        self.hw_range = hw_range

        # 加载骨干网络 pvt_v2_b2，其输出通道数分别为 [64, 128, 320, 512]
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        # 预训练模型的路径
        abspath = os.path.abspath(__file__)  # 获取所执行脚本的绝对路径
        proj_path = os.path.dirname(abspath)  # 获取父级路径
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

        # ARConv 模块，用于特征增强
        self.ARConv1 = ARConv(inc=64, outc=64)
        self.ARConv2 = ARConv(inc=128, outc=128)
        self.ARConv3 = ARConv(inc=320, outc=320)
        self.ARConv4 = ARConv(inc=512, outc=512)

        # 输入尺寸为 3x352x352
        # 通道归一化层，将输入通道数转换为指定的通道数
        self.ChannelNormalization_1 = BasicConv2d(64, channel, 3, 1, 1)  # 64x88x88->32x88x88
        self.ChannelNormalization_2 = BasicConv2d(128, channel, 3, 1, 1)  # 128x44x44->32x44x44
        self.ChannelNormalization_3 = BasicConv2d(320, channel, 3, 1, 1)  # 320x22x22->32x22x22
        self.ChannelNormalization_4 = BasicConv2d(512, channel, 3, 1, 1)  # 512x11x11->32x11x11

        # ACmix 模块，用于特征增强
        self.ACmix1 = ACmix(64, 64)
        self.ACmix2 = ACmix(128, 128)
        self.ACmix3 = ACmix(320, 320)
        self.ACmix4 = ACmix(512, 512)

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
        x1 = pvt[0]  # 64x88x88
        x2 = pvt[1]  # 128x44x44
        x3 = pvt[2]  # 320x22x22
        x4 = pvt[3]  # 512x11x11

        # 通过 ARConv 模块对特征图进行增强
        x1_ar = self.ARConv1(x1, self.epoch, self.hw_range)
        x2_ar = self.ARConv2(x2, self.epoch, self.hw_range)
        x3_ar = self.ARConv3(x3, self.epoch, self.hw_range)
        x4_ar = self.ARConv4(x4, self.epoch, self.hw_range)

        # 通过 ACmix 模块对 ARConv 处理后的特征图进行增强
        x1_dea = self.ACmix1(x1_ar)
        x2_dea = self.ACmix2(x2_ar)
        x3_dea = self.ACmix3(x3_ar)
        x4_dea = self.ACmix4(x4_ar)

        # 这里可以根据需要添加额外的处理逻辑，目前直接使用增强后的特征图
        x1_all = x1_dea
        x2_all = x2_dea
        x3_all = x3_dea
        x4_all = x4_dea

        # 通过通道归一化层将特征图的通道数统一
        x1_nor = self.ChannelNormalization_1(x1_all)  # 32x88x88
        x2_nor = self.ChannelNormalization_2(x2_all)  # 32x44x44
        x3_nor = self.ChannelNormalization_3(x3_all)  # 32x22x22
        x4_nor = self.ChannelNormalization_4(x4_all)  # 32x11x11

        # 通过解码器进行解码，得到预测结果
        prediction = self.Decoder(x4_nor, x3_nor, x2_nor, x1_nor)

        # 返回原始预测结果和经过 Sigmoid 激活后的预测结果
        return prediction, self.sigmoid(prediction)


if __name__ == '__main__':
    # 实例化 DBANet 模型
    model = DBANet()
    # 打印模型的结构
    print(model)