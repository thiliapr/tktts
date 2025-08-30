# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import math
from typing import NamedTuple, Optional
import torch
from torch import nn
from torch.nn import functional as F


class ScaleNorm(nn.Module):
    """
    实现了一个缩放归一化的模块。该模块根据输入的张量 `x` 计算其L2范数，并用可学习的缩放因子 `g` 对其进行缩放归一化。用于增强模型的稳定性和学习能力。

    工作流程如下：
        1. 输入张量 `x` 会计算其在最后一个维度上的L2范数。
        2. 通过一个可学习的参数 `g` 对输入进行缩放。
        3. 输出结果是经过缩放的归一化张量。

    Args:
        dim: 用于初始化缩放因子 `g` 的维度。

    Returns:
        返回缩放归一化后的张量。

    Examples:
        # 示例代码：
        scale_norm = ScaleNorm(dim=256)
        x = torch.randn(10, 256)  # 假设x的维度是[10, 256]
        output = scale_norm(x)  # 返回经过缩放归一化后的张量
    """

    def __init__(self, dim: int, device: Optional[torch.device] = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, device=device) * (dim ** 0.5))  # 可学习的缩放因子，初始化为dim的平方根

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(min=torch.finfo(x.dtype).eps)  # 计算L2范数并防止为零
        return self.scale * x / norm  # 对输入张量进行缩放归一化


class MultiheadAttention(nn.Module):
    """
    多头自注意力机制模块，实现基于旋转位置编码(RoPE)的自注意力计算。

    本模块主要工作流程：
    1. 通过线性投影将输入转换为查询(Q)、键(K)和值(V)张量
    2. 对Q和K应用旋转位置编码(RoPE)注入位置信息
    3. 计算缩放点积注意力，支持填充掩码
    4. 合并多头输出并通过线性投影得到最终结果

    Inputs:
        qkv: 查询、键值共享张量；由于使用的是自注意力，所以假设查询、键、值相同，形状为 (batch_size, seq_len_q, dim_model)
        padding_mask: 填充掩码，形状为 (batch_size, seq_len)，填充位置为 True，非填充位置为 False

    Outputs:
        注意力输出张量，形状为 (batch_size, seq_len, dim_model)
    """

    def __init__(self, dim_head: int, num_heads: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout
        dim_model = dim_head * num_heads  # 总模型维度 = 头数 * 每头维度

        # 查询、键值投影矩阵
        self.qkv_proj = nn.Linear(dim_model, dim_model * 3, device=device)

        # 输出投影矩阵，将多头输出合并回原始维度
        self.out_proj = nn.Linear(dim_model, dim_model, device=device)

        # RoPE 旋转频率
        self.inv_freq = nn.Buffer(1.0 / (10000 ** (torch.arange(0, dim_head, 2, device=device) / dim_head)), persistent=False)

        # freqs_cis 缓存
        self.freqs_cis_cache = nn.Buffer(torch.empty(0, dim_head // 2, device=device), persistent=False)

        # 使用 Xavier 均匀分布初始化查询、键、值、投影权重
        for module in [self.qkv_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def apply_rope(self, x: torch.Tensor) -> torch.Tensor:
        """
        应用旋转位置编码(RoPE)到输入张量。

        旋转位置编码通过复数乘法实现位置信息的注入：
        1. 将输入张量的最后两个维度视为复数对(实部和虚部)
        2. 生成与位置相关的旋转复数向量
        3. 通过复数乘法实现旋转操作
        4. 将旋转后的复数转换回实数表示

        Args:
            x: 输入张量，形状为 [batch, num_heads, seq_len, dim_head]

        Returns:
            应用旋转位置编码后的张量，形状与输入相同
        """
        # 计算需要应用RoPE的序列长度
        required_seq_len = x.size(2)

        # 检查并更新旋转频率缓存
        current_cache_len = self.freqs_cis_cache.size(0)
        if current_cache_len < required_seq_len:
            # 生成缺失位置的时间索引
            new_positions = torch.arange(
                current_cache_len,
                required_seq_len,
                device=self.inv_freq.device
            )
            # 计算新位置的旋转频率
            new_freqs = torch.outer(new_positions, self.inv_freq)
            # 转换为复数形式 (cosθ + i·sinθ)
            new_cis = torch.polar(torch.ones_like(new_freqs), new_freqs)
            # 更新缓存
            self.freqs_cis_cache = torch.cat([self.freqs_cis_cache, new_cis], dim=0)

        # 获取当前序列所需的旋转频率
        freqs_cis = self.freqs_cis_cache[:required_seq_len]

        # 将最后维度重塑为复数对 (..., dim_head//2, 2)
        # 这里转换为 float 是因为半精度复数运算不被支持
        complex_shape = x.shape[:-1] + (-1, 2)
        complex_pairs = x.float().reshape(complex_shape)

        # 转换为复数张量
        complex_tensor = torch.view_as_complex(complex_pairs)

        # 调整旋转频率形状以匹配输入 (添加批量和头维度)
        freqs_cis = freqs_cis.view(1, 1, -1, freqs_cis.shape[-1])

        # 应用旋转（复数乘法）
        rotated_complex = complex_tensor * freqs_cis

        # 转换回实数表示
        rotated_real = torch.view_as_real(rotated_complex)

        # 展平最后两个维度 (..., dim_head//2, 2) -> (..., dim_head)
        rotated_output = rotated_real.flatten(-2).to(dtype=x.dtype)
        return rotated_output

    def forward(
        self,
        qkv: torch.Tensor,
        padding_mask: torch.BoolTensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = qkv.shape

        # 计算查询、键值投影 [batch, seq_len, dim_model]
        queries, keys, values = self.qkv_proj(qkv).chunk(3, dim=-1)  # 分割为 Q, K, V

        # 调整查询、键、值形状为 [batch, seq_len, num_heads, dim_head]，并重排维度为 PyTorch 注意力要求的形状 [batch, heads, seq_len, dim_head]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        keys = keys.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        values = values.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # 应用 RoPE 到 Q/K
        queries = self.apply_rope(queries)
        keys = self.apply_rope(keys)

        # 计算填充掩码，形状为 [batch, seq_len, seq_len]
        attn_mask = padding_mask.unsqueeze(1) | padding_mask.unsqueeze(2)

        # 显式转化为 float 类型掩码，避免版本兼容性问题
        # 比如在大部分包（transformers 等），True 表示填充
        # 但在 torch.nn.functional.scaled_dot_product_attention 中，True 表示允许注意力
        attn_mask = torch.where(attn_mask, -torch.inf, 0.0)
        
        # 扩展掩码以匹配多头注意力的要求，形状为 [batch, 1, seq_len, seq_len]
        attn_mask = attn_mask.unsqueeze(1)

        # 计算缩放点积注意力
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=(self.dropout_rate if self.training else 0.),
        )

        # 合并多头输出 (batch, seq_len, dim_model)
        return self.out_proj(attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1))


class Conv(nn.Module):
    """
    一维卷积层，自动处理输入张量的维度转换
    该类继承自 nn.Module，封装了 nn.Conv1d 并自动设置 padding 为`same`模式
    在保持序列长度不变的前提下进行一维卷积操作

    工作流程：
    1. 在初始化时自动设置卷积层的padding模式为`same`
    2. 在前向传播时自动转换输入张量维度以适应 Conv1d 的要求
    3. 执行卷积操作后再将维度转换回原始格式

    Inputs:
        x: 输入张量，形状为 [batch_size, seq_len, in_channels]

    Outputs:
        输出张量，形状为 [batch_size, seq_len, out_channels]
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, *args, **kwargs):
        super().__init__()
        kwargs["padding"] = "same"
        self.model = nn.Conv1d(in_channels, out_channels, kernel_size, *args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)  # 转换为 [batch_size, in_channels, seq_len]
        x = self.model(x)  # 调用父类的 forward 方法，[batch_size, out_channels, seq_len]
        x = x.transpose(1, 2)  # 恢复为 [batch_size, seq_len, out_channels]
        return x  # 返回处理后的张量


class FFTBlock(nn.Module):
    """
    FFTBlock 是一个基于多头注意力和前馈网络的模块，主要用于处理序列数据。它包含自注意力机制和前馈网络，并使用缩放归一化来增强模型的稳定性。

    工作流程如下：
        1. 输入张量通过多头注意力机制进行处理，计算注意力输出。
        2. 将注意力输出与输入张量相加，并应用缩放归一化。
        3. 输入张量经过前馈网络处理，得到新的表示。
        4. 将前馈网络输出与注意力输出相加，并再次应用缩放归一化。

    Args:
        dim_head: 每个注意力头的维度。
        num_heads: 注意力头的数量。
        dim_feedforward: 前馈网络的隐藏层维度。
        conv1_kernel_size: 前馈网络中第一个卷积层的内核大小。
        conv2_kernel_size: 前馈网络中第二个卷积层的内核大小。
        dropout: Dropout 概率，用于防止过拟合。
        device: 可选的设备参数，用于指定模型运行的设备。

    Inputs:
        x: 输入张量，形状为 (batch_size, seq_len, dim_model)
        padding_mask: 填充掩码，形状为 (batch_size, seq_len)，用于指示哪些位置是填充的。

    Outputs:
        返回处理后的张量，形状与输入相同。
    """

    def __init__(self, dim_head: int, num_heads: int, dim_feedforward: int, conv1_kernel_size: int, conv2_kernel_size: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        dim_model = dim_head * num_heads  # 总模型维度

        # 自注意力和前馈网络
        self.attention = MultiheadAttention(dim_head, num_heads, dropout, device=device)
        self.conv1 = Conv(dim_model, dim_feedforward, conv1_kernel_size, device=device)
        self.conv2 = Conv(dim_feedforward, dim_model, conv2_kernel_size, device=device)

        # 归一化和缩放
        self.attention_norm = ScaleNorm(dim_model, device=device)
        self.feedforward_norm = ScaleNorm(dim_model, device=device)
        self.attention_scale = nn.Parameter(torch.zeros(1)).to(device)
        self.feedforward_scale = nn.Parameter(torch.zeros(1)).to(device)

        # Dropout 层
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(module.model.weight)
            nn.init.zeros_(module.model.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        # 多头注意力计算
        x = x + self.dropout(self.attention(self.attention_norm(x), padding_mask) * self.attention_scale)

        # 前馈网络计算
        res = x  # 保存残差连接

        # 归一化
        x = self.feedforward_norm(x)

        # 填充位置置零，防止填充位置通过卷积污染正常位置
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 前馈网络第一层卷积
        x = F.silu(self.conv1(x))

        # 由于经过一次卷积，填充位置已经被正常位置污染（变得非零了），所以需要重新置零
        # 具体来说，由于卷积操作会考虑邻近位置的值，如果邻近位置是非零的，那么填充位置经过卷积后也会变成非零
        # 虽然填充位置已经是零，但填充位置旁的有效位置非零，通过卷积核的加权和，填充位置就会变成非零
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 第二次卷积后不填充位置置零是因为除了卷积层需要手动置零，其他层都不用管
        x = self.dropout(self.conv2(x) * self.feedforward_scale)

        # 残差连接
        return res + x


class VariancePredictor(nn.Module):
    """
    VariancePredictor 是一个用于预测时长、音高或能量的卷积神经网络模块。
    它由多个卷积层组成，每个卷积层后面跟着缩放归一化、SiLU 激活函数和 Dropout 层。

    该模块的主要工作流程如下：
        1. 输入张量通过一系列卷积层进行处理。
        2. 每个卷积层后应用缩放归一化以增强模型的稳定性。
        3. 使用 SiLU 激活函数引入非线性。
        4. 应用 Dropout 层以防止过拟合。
        5. 最终通过一个卷积层输出预测结果。

    Args:
        dim_model: 输入张量的维度。
        kernel_size: 卷积层的内核大小。
        dropout: Dropout 概率，用于防止过拟合。
        device: 可选的设备参数，用于指定模型运行的设备。

    Inputs:
        x: 输入张量，形状为 (batch_size, seq_len, dim_model)
        padding_mask: 填充掩码，形状为 (batch_size, seq_len)，填充位置为 True，非填充位置为 False

    Outputs:
        返回预测的时长、音高或能量，形状为 (batch_size, seq_len)
    """

    def __init__(self, dim_model: int, kernel_size: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        self.conv1 = Conv(dim_model, dim_model, kernel_size, device=device)
        self.conv2 = Conv(dim_model, dim_model, kernel_size, device=device)
        self.output_layer = nn.Linear(dim_model, 1, device=device)
        self.norm1 = ScaleNorm(dim_model, device=device)
        self.norm2 = ScaleNorm(dim_model, device=device)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for module in [self.conv1, self.conv2]:
            nn.init.xavier_uniform_(module.model.weight)
            nn.init.zeros_(module.model.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        # 第一层卷积和归一化
        x = x + self.dropout(F.silu(self.conv1(self.norm1(x).masked_fill(padding_mask.unsqueeze(-1), 0))))

        # 第二层卷积和归一化
        x = x + self.dropout(F.silu(self.conv2(self.norm2(x).masked_fill(padding_mask.unsqueeze(-1), 0))))

        # 最终输出层，形状为 [batch_size, seq_len, 1] => [batch_size, seq_len]
        x = self.output_layer(x).squeeze(-1)

        # 再次对填充位置置零，防止填充位置输出非零结果影响音高、能量或时长预测
        x = x.masked_fill(padding_mask, 0)
        return x


class DifferentiableLengthRegulator(nn.Module):
    """
    可微分长度调节器，用于根据持续时间动态调整特征序列长度。
    通过软对齐方式实现特征扩展，保持模型可微分性。

    工作原理：
    1. 计算输入特征的累积持续时间
    2. 根据最大持续时间确定输出帧数
    3. 使用softmax权重计算每帧对应的特征加权和
    4. 通过温度系数控制对齐的尖锐程度

    Args:
        temperature: 控制对齐尖锐程度的温度参数，值越小对齐越尖锐（接近one-hot），值越大对齐越平滑

    Inputs:
        features: 输入特征序列 [batch_size, seq_len, feature_dim]
        durations: 每个特征的持续时间 [batch_size, seq_len]
        padding_mask: 填充掩码，形状为 [batch_size, seq_len]，填充位置为 True，非填充位置为 False。

    Outputs:
        扩展后的特征序列 [batch_size, total_frames, feature_dim]
    """

    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
        
    def forward(self, features: torch.Tensor, durations: torch.Tensor, padding_mask: torch.BoolTensor):
        # 计算每个时间步的累积持续时间 [batch_size, seq_len]
        cum_durations = torch.cumsum(durations, dim=1)

        # 计算最大持续时间并确定总帧数（向上取整）
        total_frames = int(torch.ceil(cum_durations[:, -1].max()).item())

        # 创建帧中心位置网格 [1, total_frames]
        # 使用0.5偏移使帧中心对齐
        frame_centers = torch.arange(total_frames, device=features.device).float()[None] + 0.5

        # 计算每帧与各特征位置的距离 [batch_size, total_frames, seq_len]
        # 使用广播机制自动扩展维度
        frame_distances = frame_centers[:, :, None] - cum_durations[:, None, :]

        # 计算权重，使用温度参数控制分布形状 [batch_size, total_frames, seq_len]
        logits = -torch.abs(frame_distances) / self.temperature

        # 应用填充掩码
        logits = logits.masked_fill(padding_mask[:, None, :].expand_as(logits), -torch.inf)

        # 计算归一化权重 [batch_size, total_frames, seq_len]
        weights = torch.softmax(logits, dim=-1)

        # 通过加权求和扩展特征 [batch_size, total_frames, feature_dim]
        expanded_features = torch.einsum("btl,bld->btd", weights, features)
        return expanded_features


class PostNet(nn.Module):
    """
    后处理网络，用于语音合成中的梅尔频谱图精细化处理。
    该网络由多个卷积层和缩放归一化层组成，使用 tanh 激活函数和 dropout 正则化。
    通过卷积操作增强频谱图的局部特征表达，并通过缩放归一化稳定训练过程。

    工作流程：
        1. 对输入频谱图进行掩码处理，将填充位置置零
        2. 通过多个卷积+归一化+激活层序列处理特征
        3. 最终通过一个卷积层输出精细化后的频谱图

    Inputs:
        x: 输入梅尔频谱图，形状为 [batch_size, seq_len, num_mels]
        padding_mask: 填充掩码，True 表示填充位置

    Outputs:
        精细化后的梅尔频谱图，形状与输入相同 [batch_size, seq_len, num_mels]
    """

    def __init__(self, num_mels: int, hidden_dim: int, kernel_size: int, num_layers: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        self.layers = nn.ModuleList()

        # 添加第一个卷积层（输入层）
        self.layers.append(nn.Sequential(Conv(num_mels, hidden_dim, kernel_size, device=device), ScaleNorm(hidden_dim, device=device)))
    
        # 添加 num_layers - 2 个隐藏层
        self.layers.extend(nn.Sequential(Conv(hidden_dim, hidden_dim, kernel_size, device=device), ScaleNorm(hidden_dim, device=device)) for _ in range(num_layers - 2))

        # 输出层卷积，恢复原始维度
        self.output_conv = Conv(hidden_dim, num_mels, kernel_size, device=device)
        self.dropout = nn.Dropout(dropout)

        # 初始化权重
        for layer in [*[seq[0] for seq in self.layers], self.output_conv]:
            nn.init.xavier_uniform_(layer.model.weight)
            nn.init.zeros_(layer.model.bias)

    def forward(self, x: torch.Tensor, padding_mask: torch.BoolTensor) -> torch.Tensor:
        # 对填充位置进行掩码处理
        x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 通过输入层、所有隐藏层
        for layer in self.layers:
            x = self.dropout(F.tanh(layer(x)))
            # 再次应用掩码，确保填充位置不影响后续计算
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 最终输出层
        return self.dropout(self.output_conv(x))


class FastSpeech2Config(NamedTuple):
    """
    FastSpeech2 的配置类，包含模型的超参数设置。

    Attributes:
        vocab_size: 词汇表大小
        num_tags: 标签数量，用于控制生成
        num_mels: 梅尔频谱的维度
        num_heads: 注意力头的数量
        dim_head: 每个注意力头的维度
        dim_feedforward: 前馈网络的隐藏层维度
        fft_conv1_kernel_size: 第一个 FFT 卷积层的卷积核大小
        fft_conv2_kernel_size: 第二个 FFT 卷积层的卷积核大小
        predictor_kernel_size: 预测器的卷积核大小
        variance_bins: 变异性预测的 bins 数量
        postnet_hidden_dim: 后处理网络的隐藏层维度
        postnet_kernel_size: 后处理网络中卷积层的卷积核大小
        num_encoder_layers: 编码器层的数量
        num_decoder_layers: 解码器层的数量
        num_postnet_layers: 后处理网络中卷积层的数量
    """
    vocab_size: int
    num_tags: int
    num_mels: int
    num_heads: int
    dim_head: int
    dim_feedforward: int
    fft_conv1_kernel_size: int
    fft_conv2_kernel_size: int
    predictor_kernel_size: int
    variance_bins: int
    postnet_hidden_dim: int
    postnet_kernel_size: int
    num_encoder_layers: int
    num_decoder_layers: int
    num_postnet_layers: int


class FastSpeech2(nn.Module):
    """
    FastSpeech2 模型实现，基于 FastSpeech2Config 配置类。
    该模型包含词嵌入、标签嵌入、编码器、解码器、预测器和梅尔频谱预测器等组件。

    工作流程如下：
        1. 输入文本通过词嵌入层转换为嵌入表示。
        2. 如果提供了标签，则将标签嵌入并与文本嵌入相加。
        3. 嵌入表示通过编码器进行处理，生成上下文表示。
        4. 使用预测器预测音高、能量和时长。
        5. 将音高和能量作为附加特征添加到上下文表示中。
        6. 使用长度调节器调整序列长度。
        7. 解码器处理调整后的表示，生成梅尔频谱预测。
        8. 使用 PostNet 精细化梅尔频谱，并进行残差连接。

    Args:
        config: FastSpeech2Config 配置对象，包含模型的超参数设置。
        dropout: Dropout 概率，用于防止过拟合。
        device: 设备参数，用于指定模型运行的设备。

    Inputs:
        text: 输入文本的张量，形状为 (batch_size, text_len)。
        positive_prompt: 正向提示标签张量，形状为 (batch_size, num_positive_prompts)。
        negative_prompt: 负向提示标签张量，形状为 (batch_size, num_negative_prompts)。
        text_padding_mask: 文本填充掩码，形状同 text。填充位置为 True，非填充位置为 False。
        positive_prompt_mask: 正向提示标签填充掩码，形状同 positive_prompt。
        negative_prompt_mask: 负向提示标签填充掩码，形状同 negative_prompt。
        duration_sum_target: 总时长目标张量，形状为 (batch_size)。
        pitch_target: 音高目标张量，形状为 (batch_size, audio_len)。
        energy_target: 能量目标张量，形状为 (batch_size, audio_len)。

    Outputs:
        返回一个元组，包含后处理网络输出的梅尔频谱预测、原始梅尔频谱预测、时长预测、音高预测、能量预测和音频预测填充掩码，形状分别为：
        - postnet_prediction: (batch_size, audio_len, num_mels)
        - mel_prediction: (batch_size, audio_len, num_mels)
        - mel_prediction: (batch_size, audio_len, num_mels)
        - duration_prediction: (batch_size, audio_len)
        - pitch_prediction: (batch_size, audio_len)
        - energy_prediction: (batch_size, audio_len)
        - audio_padding_mask: (batch_size)
    """

    def __init__(self, config: FastSpeech2Config, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        self.dim_model = config.dim_head * config.num_heads  # 总模型维度
        self.variance_bins = config.variance_bins

        # 词嵌入
        self.embedding = nn.Embedding(config.vocab_size, self.dim_model, device=device)
        self.tag_embedding = nn.Embedding(config.num_tags, self.dim_model, device=device)

        # 编码器、解码器
        layer = FFTBlock(config.dim_head, config.num_heads, config.dim_feedforward, config.fft_conv1_kernel_size, config.fft_conv2_kernel_size, dropout, device)
        self.encoder = nn.ModuleList(copy.deepcopy(layer) for _ in range(config.num_encoder_layers))
        self.decoder = nn.ModuleList(copy.deepcopy(layer) for _ in range(config.num_decoder_layers))

        # 预测器
        self.duration_predictor = VariancePredictor(self.dim_model, config.predictor_kernel_size, dropout, device)
        self.pitch_predictor = VariancePredictor(self.dim_model, config.predictor_kernel_size, dropout, device)
        self.energy_predictor = VariancePredictor(self.dim_model, config.predictor_kernel_size, dropout, device)

        # 音高和能量嵌入
        self.pitch_embedding = nn.Embedding(config.variance_bins, self.dim_model, device=device)
        self.energy_embedding = nn.Embedding(config.variance_bins, self.dim_model, device=device)

        # 长度调节器
        self.length_regulator = DifferentiableLengthRegulator()

        # 梅尔频谱预测器
        self.mel_predictor = nn.Linear(self.dim_model, config.num_mels, device=device)

        # 后处理网络
        self.postnet = PostNet(config.num_mels, config.postnet_hidden_dim, config.postnet_kernel_size, config.num_postnet_layers, dropout, device)

        # 后处理网络缩放因子
        self.postnet_scale = nn.Parameter(torch.zeros(1)).to(device)

        # 初始化权重
        nn.init.zeros_(self.mel_predictor.bias)
        for module in [self.embedding, self.tag_embedding, self.pitch_embedding, self.energy_embedding, self.mel_predictor]:
            nn.init.xavier_uniform_(module.weight)

    def prompt_embedding(self, prompt: torch.LongTensor, padding_mask: Optional[torch.BoolTensor]):
        """
        获取提示词的嵌入表示，并对填充部分进行处理。

        工作流程如下：
            1. 通过标签嵌入层获取每个标签的嵌入表示。
            2. 将填充部分的嵌入值置为零。
            3. 计算有效标签的数量。
            4. 对有效标签的嵌入求平均，得到最终的提示词嵌入。

        Args:
            prompt: 正向提示标签的张量，形状为 (batch_size, num_tags)。
            padding_mask: 正向提示标签的填充掩码，形状同 prompt。

        Returns:
            返回提示词的平均嵌入表示，形状为 (batch_size, 1, dim_model)。
        """
        # 获取提示词每个标签的嵌入
        tag_embedding = self.tag_embedding(prompt)  # [batch_size, num_tags, dim_model]

        # 将填充部分置零
        tag_embedding = tag_embedding.masked_fill(padding_mask.unsqueeze(-1), 0)

        # 计算有效标签数量
        valid_tag_count = (~padding_mask).sum(dim=1, keepdim=True)  # [batch_size, 1]

        # 对有效标签嵌入求平均
        tag_mean_embedding = tag_embedding.sum(dim=1) / valid_tag_count.clamp(min=1)  # [batch_size, dim_model]

        # 给提示词增加序列维度
        return tag_mean_embedding.unsqueeze(1)  # [batch_size, 1, dim_model]

    def forward(
        self,
        text: torch.LongTensor,
        positive_prompt: Optional[torch.LongTensor] = None,
        negative_prompt: Optional[torch.LongTensor] = None,
        text_padding_mask: Optional[torch.BoolTensor] = None,
        positive_prompt_mask: Optional[torch.BoolTensor] = None,
        negative_prompt_mask: Optional[torch.BoolTensor] = None,
        duration_sum_target: Optional[torch.Tensor] = None,
        pitch_target: Optional[torch.Tensor] = None,
        energy_target: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor]:
        # 如果没有提供填充掩码，则创建一个全为 False 的掩码
        if text_padding_mask is None:
            text_padding_mask = torch.zeros_like(text, device=text.device, dtype=bool)  # [batch_size, text_len]
        if positive_prompt is not None and positive_prompt_mask is None:
            positive_prompt_mask = torch.zeros_like(positive_prompt, device=text.device, dtype=bool)  # [batch_size, num_pos_tags]
        if negative_prompt is not None and negative_prompt_mask is None:
            negative_prompt_mask = torch.zeros_like(negative_prompt, device=text.device, dtype=bool)  # [batch_size, num_neg_tags]

        # 词嵌入
        x = self.embedding(text) * math.sqrt(self.dim_model)

        # 标签嵌入
        if positive_prompt is not None:
            x = x + self.prompt_embedding(positive_prompt, positive_prompt_mask)
        if negative_prompt is not None:
            x = x - self.prompt_embedding(negative_prompt, negative_prompt_mask)

        # 编码器
        for layer in self.encoder:
            x = layer(x, text_padding_mask)

        # 预测时长
        duration_prediction = self.duration_predictor(x, text_padding_mask)  # [batch_size, audio_len]

        # 防止时长小于零
        duration = duration_prediction.clamp(min=0)

        # 对每个样本，如果总持续时间为 0，则将每个元素设为 1，避免后续除零错误
        duration += (duration.sum(dim=1, keepdim=True) == 0).to(dtype=duration.dtype)

        # 调节时长，使其总和与目标值相同
        if duration_sum_target is not None:
            duration, duration_sum_target = duration.float(), duration_sum_target.float()  # 转化为 float，避免精度丢失导致后面计算 duration_sum_pred != duration_sum_target
            duration *= (duration_sum_target.unsqueeze(1) / duration.sum(dim=1, keepdim=True))  # 计算缩放，使 duration_sum_pred 等于 duration_sum_target
            duration = duration.to(dtype=x.dtype)  # 转换回原来精度

        # 长度调节
        x = self.length_regulator(x, duration, text_padding_mask)  # [batch_size, dim_model, audio_len]

        # 实践证明，duration.sum(dim=1).ceil() != duration_sum_target 是经常的事，所以要填充或截断
        if pitch_target is not None:
            # 如果预测长度小于目标长度，则用零填充，否则截断到目标长度
            if padding_len := max(pitch_target.size(1) - x.size(1), 0):
                x = torch.cat([x, torch.zeros(x.size(0), padding_len, x.size(2), device=x.device)], dim=1)
            x = x[:, :pitch_target.size(1)]

        # 生成音频填充掩码
        audio_padding_mask = duration.sum(dim=1).ceil().unsqueeze(1) <= torch.arange(x.size(1), device=x.device).unsqueeze(0)  # [batch_size, audio_len]

        # 预测音高和能量
        pitch_prediction = self.pitch_predictor(x, audio_padding_mask)  # [batch_size, text_len]
        energy_prediction = self.energy_predictor(x, audio_padding_mask)  # [batch_size, audio_len]

        # 使用目标值替代预测值（如果提供）
        # 这是为了在训练时使用真实值进行音高、调节（即教师强制），然后返回预测值以进行损失计算
        pitch = pitch_prediction if pitch_target is None else pitch_target
        energy = energy_prediction if energy_target is None else energy_target

        # 归一化音高，并离散化
        pitch = ((pitch - pitch.min()) / (pitch.max() - pitch.min() + torch.finfo(pitch.dtype).eps) * (self.variance_bins - 1)).to(dtype=int)
        energy = ((energy - energy.min()) / (energy.max() - energy.min() + torch.finfo(energy.dtype).eps) * (self.variance_bins - 1)).to(dtype=int)

        # 将音高和能量作为附加特征添加到解码器输入中
        x = x + self.pitch_embedding(pitch) + self.energy_embedding(energy)

        # 解码器
        for layer in self.decoder:
            x = layer(x, audio_padding_mask)

        # 梅尔频谱预测
        mel_prediction = self.mel_predictor(x)  # [batch_size, audio_len, num_mels]

        # 通过后处理网络
        postnet_prediction = mel_prediction + self.postnet(mel_prediction, audio_padding_mask) * self.postnet_scale
        return postnet_prediction, mel_prediction, duration_prediction, pitch_prediction, energy_prediction, audio_padding_mask
