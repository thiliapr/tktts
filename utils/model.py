# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import copy
import math
from typing import NamedTuple, Optional, overload
import torch
from torch import nn
from torch.nn import functional as F

AttentionKVCache = tuple[torch.Tensor, torch.Tensor]
LayerKVCache = tuple[AttentionKVCache, AttentionKVCache]
NetKVCache = list[LayerKVCache]


class ScaleNorm(nn.Module):
    """
    实现了一个缩放归一化的模块。该模块根据输入的张量 `x` 计算其L2范数，并用可学习的缩放因子 `g` 对其进行缩放归一化。用于增强模型的稳定性和学习能力。

    工作流程如下：
        1. 输入张量 `x` 会计算其在最后一个维度上的L2范数。
        2. 通过一个可学习的参数 `g` 对输入进行缩放。
        3. 输出结果是经过缩放的归一化张量。

    Args:
        dim: 用于初始化缩放因子 `g` 的维度。
        eps: 防止除零的常数（默认为1e-5）。

    Returns:
        返回缩放归一化后的张量。

    Examples:
        # 示例代码：
        scale_norm = ScaleNorm(dim=256)
        x = torch.randn(10, 256)  # 假设x的维度是[10, 256]
        output = scale_norm(x)  # 返回经过缩放归一化后的张量
    """

    def __init__(self, dim: int, eps: float = 1e-5, device: Optional[torch.device] = None):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1, device=device) * (dim ** 0.5))  # 可学习的缩放因子，初始化为dim的平方根
        self.eps = eps  # 避免除零错误的小常数

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp(min=self.eps)  # 计算L2范数并防止为零
        return self.scale * x / norm  # 对输入张量进行缩放归一化


class MultiheadAttention(nn.Module):
    """
    多头注意力机制模块，实现基于旋转位置编码(RoPE)的自注意力和交叉注意力计算。

    本模块主要工作流程：
    1. 通过线性投影将输入转换为查询(Q)、键(K)和值(V)张量
    2. 对Q和K应用旋转位置编码(RoPE)注入位置信息
    3. 计算缩放点积注意力，支持因果掩码和填充掩码
    4. 合并多头输出并通过线性投影得到最终结果

    支持以下特性：
    - KV缓存机制，用于高效的自回归生成
    - 增量解码时的位置编码缓存
    - 可配置的注意力头数和每头维度
    - 可选的dropout正则化

    Inputs:
        queries: 查询张量，形状为 (batch_size, seq_len_q, dim_model)
        keys_values: 键值张量，形状为 (batch_size, seq_len_k, dim_model)，在自注意力模式下与queries相同。kv_cache 不为 None 时，keys_values 可为 None
        padding_mask: 填充掩码，形状为 (batch_size, seq_len)
        kv_cache: 键值缓存元组，包含 (K_cache, V_cache)
        queries_rope_offset: 查询序列在RoPE位置编码中的起始偏移量
        keys_rope_offset: 键序列在RoPE位置编码中的起始偏移量
        is_causal: 是否使用因果注意力掩码

    Outputs:
        元组包含:
        - 注意力输出张量，形状为(batch_size, seq_len, dim_model)
        - 更新后的键值缓存(KV cache)
    """

    def __init__(self, dim_head: int, num_heads: int, dropout: float = 0., device: Optional[torch.device] = None):
        super().__init__()
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.dropout_rate = dropout
        dim_model = dim_head * num_heads  # 总模型维度 = 头数 * 每头维度

        # 查询、键值投影矩阵
        self.queries_proj = nn.Linear(dim_model, dim_model, device=device)
        self.kv_proj = nn.Linear(dim_model, dim_model * 2, device=device)

        # 输出投影矩阵，将多头输出合并回原始维度
        self.out_proj = nn.Linear(dim_model, dim_model, device=device)

        # RoPE 旋转频率
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim_head, 2, device=device) / dim_head))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # freqs_cis 缓存
        self.freqs_cis_cache: torch.Tensor
        self.register_buffer("freqs_cis_cache", torch.empty(0, dim_head // 2, device=device), persistent=False)

        # 使用 Xavier 均匀分布初始化查询、键、值、投影权重
        for module in [self.queries_proj, self.kv_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def apply_rope(self, x: torch.Tensor, rope_offset: int = 0) -> torch.Tensor:
        """
        应用旋转位置编码(RoPE)到输入张量。

        旋转位置编码通过复数乘法实现位置信息的注入：
        1. 将输入张量的最后两个维度视为复数对(实部和虚部)
        2. 生成与位置相关的旋转复数向量
        3. 通过复数乘法实现旋转操作
        4. 将旋转后的复数转换回实数表示

        该方法支持增量计算：
        - 维护旋转频率缓存(freqs_cis_cache)避免重复计算
        - 当序列长度超过缓存大小时自动扩展缓存
        - 支持从指定位置开始应用旋转编码

        Args:
            x: 输入张量，形状为 [batch, num_heads, seq_len, dim_head]
            rope_offset: 序列在位置编码中的起始偏移量，用于增量解码

        Returns:
            应用旋转位置编码后的张量，形状与输入相同
        """
        # 计算需要应用RoPE的序列长度
        required_seq_len = x.size(2)

        # 检查并更新旋转频率缓存
        current_cache_len = self.freqs_cis_cache.size(0)
        if current_cache_len < rope_offset + required_seq_len:
            # 生成缺失位置的时间索引
            new_positions = torch.arange(
                current_cache_len,
                rope_offset + required_seq_len,
                device=self.inv_freq.device
            )
            # 计算新位置的旋转频率
            new_freqs = torch.outer(new_positions, self.inv_freq)
            # 转换为复数形式 (cosθ + i·sinθ)
            new_cis = torch.polar(torch.ones_like(new_freqs), new_freqs)
            # 更新缓存
            self.freqs_cis_cache = torch.cat([self.freqs_cis_cache, new_cis], dim=0)

        # 获取当前序列所需的旋转频率
        freqs_cis = self.freqs_cis_cache[rope_offset:rope_offset + required_seq_len]

        # 将最后维度重塑为复数对 (..., dim_head//2, 2)
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

        # 组合结果：未旋转部分 + 旋转后的部分
        return rotated_output

    @overload
    def forward(
        self,
        queries: torch.Tensor,
        keys_values: Optional[torch.Tensor],
        padding_mask: None,
        kv_cache: AttentionKVCache,
        queries_rope_offset: int,
        keys_rope_offset: int,
        is_causal: bool,
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        ...

    @overload
    def forward(
        self,
        queries: torch.Tensor,
        keys_values: torch.Tensor,
        padding_mask: Optional[torch.BoolTensor],
        kv_cache: None,
        queries_rope_offset: int,
        keys_rope_offset: int,
        is_causal: bool,
    ) -> tuple[torch.Tensor, AttentionKVCache]:
        ...

    def forward(
        self,
        queries,
        keys_values=None,
        padding_mask=None,
        kv_cache=None,
        queries_rope_offset=0,
        keys_rope_offset=0,
        is_causal=False,
    ):
        batch_size, seq_len, _ = queries.shape

        # 万恶的检查。为什么不能假设用户已经按照类型标注的去做呢？
        assert keys_values is not None or kv_cache, "至少提供 keys_values 或 kv_cache 中一个。"
        assert not is_causal or queries is keys_values, "交叉注意力使用因果掩码，你真是个小天才。"

        # 计算查询、键值投影 [batch, seq_len, dim_model]
        queries = self.queries_proj(queries)
        if keys_values is not None:
            keys, values = self.kv_proj(keys_values).chunk(2, dim=-1)

        # 调整查询、键、值形状为 [batch, seq_len, num_heads, dim_head]，并重排维度为 PyTorch 注意力要求的形状 [batch, heads, seq_len, dim_head]
        queries = queries.view(batch_size, seq_len, self.num_heads, self.dim_head).transpose(1, 2)
        if keys_values is not None:
            keys = keys.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
            values = values.view(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # 应用 RoPE 到 Q/K
        queries = self.apply_rope(queries, queries_rope_offset)
        if queries is keys_values:  # 仅在自注意力模式对 keys 进行位置编码
            keys = self.apply_rope(keys, keys_rope_offset)

        # 应用 KV Cache
        if kv_cache:
            # 拼接缓存
            if keys_values is None:
                keys, values = kv_cache
            else:
                keys = torch.cat([kv_cache[0], keys], dim=2)
                values = torch.cat([kv_cache[1], values], dim=2)

        # 处理注意力掩码
        if padding_mask is None:
            attn_mask = None
            use_builtin_causal = is_causal
        else:
            # 这里不使用内置因果掩码的原因是，F.scaled_dot_product_attention 不支持同时使用 attn_mask 和 is_causal
            use_builtin_causal = False

            if is_causal:
                causal_mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=queries.device).triu(diagonal=1)
                attn_mask = (
                    causal_mask.unsqueeze(0)
                    | padding_mask.unsqueeze(1)
                    | padding_mask.unsqueeze(2)
                )
            else:
                attn_mask = padding_mask.unsqueeze(1).expand(-1, queries.size(2), -1)

            attn_mask = attn_mask.unsqueeze(1)
            attn_mask = torch.where(attn_mask, -torch.inf, 0.)

        # 计算缩放点积注意力
        attn_output = F.scaled_dot_product_attention(
            queries, keys, values,
            attn_mask=attn_mask,
            dropout_p=(self.dropout_rate if self.training else 0.),
            is_causal=use_builtin_causal
        )

        # 合并多头输出 (batch, seq_len, dim_model)
        return self.out_proj(attn_output.transpose(1, 2).reshape(batch_size, seq_len, -1)), (keys, values)


class TransformerEncoderLayer(nn.Module):
    """
    Transformer 编码器层，实现基于Transformer的特征编码

    该层包含多头自注意力机制和前馈神经网络两个核心组件，使用残差连接和缩放归一化。
    工作流程：
      1. 输入张量首先通过注意力归一化层
      2. 计算多头自注意力并应用缩放因子
      3. 通过残差连接和Dropout合并注意力结果
      4. 处理后的张量通过前馈网络归一化层
      5. 通过前馈神经网络并应用缩放因子
      6. 再次通过残差连接和Dropout生成最终输出

    层设计特点：
      - 使用ScaleNorm代替LayerNorm增强训练稳定性
      - 引入可学习的缩放因子调整各模块贡献
      - 残差连接缓解梯度消失问题

    Args:
        num_heads: 注意力头的数量
        dim_head: 每个注意力头的维度
        dim_feedforward: 前馈网络中间层维度
        dropout: Dropout概率值，默认为0
        device: 模型参数所在的设备

    Inputs:
        x: 输入特征张量，形状为 [batch_size, seq_len, dim_model]
        padding_mask: 可选的填充掩码，形状为 [batch_size, seq_len]

    Outputs:
        编码后的特征张量，形状为[batch_size, seq_len, dim_model]

    Examples:
        >>> layer = TkTTSEncoderLayer(num_heads=4, dim_head=64, dim_feedforward=1024)
        >>> inputs = torch.randn(2, 50, 256)  # 示例输入
        >>> outputs = layer(inputs)
    """

    def __init__(
        self,
        num_heads: int,
        dim_head: int,
        dim_feedforward: int,
        dropout: float = 0.,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        dim_model = dim_head * num_heads  # 计算模型总维度

        # 初始化多头自注意力模块
        self.attention = MultiheadAttention(
            dim_head=dim_head,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )

        # 前馈神经网络定义（线性层+GELU激活）
        self.feedforward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward, device=device),
            nn.GELU(approximate="tanh"),  # 使用tanh近似的GELU
            nn.Linear(dim_feedforward, dim_model, device=device)
        )

        # 初始化归一化层
        self.attention_norm = ScaleNorm(dim_model, device=device)
        self.feedforward_norm = ScaleNorm(dim_model, device=device)

        # 初始化可学习缩放因子
        self.attention_scale = nn.Parameter(torch.zeros(1, device=device))
        self.feedforward_scale = nn.Parameter(torch.zeros(1, device=device))

        # 共享的Dropout层
        self.dropout = nn.Dropout(dropout)

        # 使用Xavier均匀分布初始化前馈网络权重
        for module in self.feedforward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 注意力分支处理（自注意力时，kv=q）
        normalized_x = self.attention_norm(x)
        attn_output, _ = self.attention(
            normalized_x, normalized_x,
            padding_mask=padding_mask,
        )

        # 应用缩放因子和残差连接
        x = x + self.dropout(attn_output * self.attention_scale)

        # 前馈网络分支处理
        ff_output = self.feedforward(self.feedforward_norm(x))

        # 应用缩放因子和残差连接
        x = x + self.dropout(ff_output * self.feedforward_scale)

        return x


class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器的单层实现，包含自注意力、交叉注意力和前馈网络三个核心组件。
    该层支持增量解码和键值缓存(KV Cache)优化，特别适用于文本到语音(TTS)等序列生成任务。

    工作流程：
    1. 输入目标序列首先通过自注意力机制处理，使用旋转位置编码和因果掩码确保自回归特性
    2. 自注意力输出与记忆序列（编码器输出）进行交叉注意力计算
    3. 交叉注意力结果通过前馈神经网络进行非线性变换
    4. 每步操作后应用残差连接、层归一化和可学习的缩放因子
    5. 支持增量解码模式，通过KV缓存避免重复计算

    Args:
        num_heads: 注意力头的数量
        dim_head: 每个注意力头的维度
        dim_feedforward: 前馈网络的隐藏层维度
        dropout: Dropout概率，用于正则化
        device: 模型运行的设备（CPU/GPU）

    Inputs:
        target: 目标序列张量，形状为 [batch_size, seq_len, dim_model]
        memory: 记忆序列（编码器输出），形状为 [batch_size, mem_len, dim_model]
        target_padding_mask: 目标序列填充掩码，形状为 [batch_size, seq_len]
        memory_padding_mask: 记忆序列填充掩码，形状为 [batch_size, mem_len]
        kv_cache: 上一时间步的键值缓存

    Outputs:
        output: 解码层输出张量，形状同输入target
        new_kv_cache: 更新后的键值缓存，用于后续时间步
    """

    def __init__(
        self,
        num_heads: int,
        dim_head: int,
        dim_feedforward: int,
        dropout: float = 0.,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        dim_model = dim_head * num_heads  # 计算模型总维度

        # 初始化多头注意力模块
        self.self_attention = MultiheadAttention(
            dim_head=dim_head,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )
        self.cross_attention = MultiheadAttention(
            dim_head=dim_head,
            num_heads=num_heads,
            dropout=dropout,
            device=device
        )

        # 前馈神经网络定义（线性层+GELU激活）
        self.feedforward = nn.Sequential(
            nn.Linear(dim_model, dim_feedforward, device=device),
            nn.GELU(approximate="tanh"),  # 使用tanh近似的GELU
            nn.Linear(dim_feedforward, dim_model, device=device)
        )

        # 初始化归一化层
        self.self_attention_norm = ScaleNorm(dim_model, device=device)
        self.cross_attention_norm = ScaleNorm(dim_model, device=device)
        self.feedforward_norm = ScaleNorm(dim_model, device=device)

        # 初始化可学习缩放因子
        self.self_attention_scale = nn.Parameter(torch.zeros(1, device=device))
        self.cross_attention_scale = nn.Parameter(torch.zeros(1, device=device))
        self.feedforward_scale = nn.Parameter(torch.zeros(1, device=device))

        # 共享的Dropout层
        self.dropout = nn.Dropout(dropout)

        # 使用Xavier均匀分布初始化前馈网络权重
        for module in self.feedforward:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    @overload
    def forward(
        self,
        target: torch.Tensor,
        memory: torch.Tensor,
        target_padding_mask: Optional[torch.Tensor],
        memory_padding_mask: Optional[torch.Tensor],
        kv_cache: None,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        ...

    @overload
    def forward(
        self,
        target: torch.Tensor,
        memory: None,
        target_padding_mask: None,
        memory_padding_mask: None,
        kv_cache: LayerKVCache,
    ) -> tuple[torch.Tensor, LayerKVCache]:
        ...

    def forward(
        self,
        target,
        memory,
        target_padding_mask=None,
        memory_padding_mask=None,
        kv_cache=None,
    ):
        # 获取位置偏移量
        rope_offset = kv_cache[0][0].size(2) if kv_cache else 0

        # 自注意力，应用缩放因子和残差连接
        normalized_target = self.self_attention_norm(target)
        attn_output, self_attn_kv_cache = self.self_attention(
            normalized_target, normalized_target,
            kv_cache=kv_cache[0] if kv_cache else None,
            padding_mask=target_padding_mask,
            queries_rope_offset=rope_offset,
            keys_rope_offset=rope_offset,
            is_causal=True
        )
        x = target + self.dropout(attn_output * self.self_attention_scale)

        # 交叉注意力
        attn_output, cross_attn_kv_cache = self.cross_attention(
            self.cross_attention_norm(x),
            memory,
            kv_cache=kv_cache[1] if kv_cache else None,
            padding_mask=memory_padding_mask,
            queries_rope_offset=rope_offset
        )
        x = x + self.dropout(attn_output * self.cross_attention_scale)

        # 前馈网络
        ff_output = self.feedforward(self.feedforward_norm(x))
        x = x + self.dropout(ff_output * self.feedforward_scale)

        return x, (self_attn_kv_cache, cross_attn_kv_cache)


class TkTTSConfig(NamedTuple):
    """
    TkTTS 的配置。

    Attributes:
        vocab_size: 词汇表大小。
        fft_length: FFT窗口长度。
        num_heads: 注意力头数量。
        dim_head: 每个注意力头的维度。
        dim_feedforward: 前馈网络的隐藏层维度。
        num_layers: Transformer En/Decoder 层的数量。
    """
    vocab_size: int
    fft_length: int
    num_heads: int
    dim_head: int
    dim_feedforward: int
    num_layers: int


class TkTTS(nn.Module):
    """
    基于Transformer的文本到语音(TTS)模型，将文本转换为组合频谱图

    工作流程：
    1. 将输入文本序列通过嵌入层转换为向量表示
    2. 使用多层编码器处理文本序列，提取高级特征
    3. 使用多层解码器处理目标序列（组合频谱），结合编码器输出
    4. 通过线性层预测组合频谱图和停止标志
    5. 支持自回归生成模式，使用KV缓存加速推理

    Inputs:
        source: 源文本序列的整数张量
        target: 目标组合频谱序列的张量
        source_padding_mask: 源序列的填充掩码
        target_padding_mask: 目标序列的填充掩码
        kv_cache: 自回归生成时的键值缓存

    Outputs:
        output:
            audio_prediction: 预测的组合频谱，形状 (batch_size, tgt_len, (fft_length // 2 + 1) * 3)
            stop_prediction: 预测的停止标志，形状 (batch_size, tgt_len)
        layers_kv_cache: 每层解码器的键值缓存（用于自回归生成）

    Examples:
        >>> model = TkTTS(config)
        >>> source = torch.randint(0, vocab_size, (batch_size, src_len))
        >>> target = torch.randn(batch_size, tgt_len, (fft_length // 2 + 1) * 3)
        >>> source_mask = torch.ones(batch_size, src_len).bool()
        >>> target_mask = torch.ones(batch_size, tgt_len).bool()
        >>> (audio_pred, stop_pred), _ = model(source, target, source_mask, target_mask)
    """

    def __init__(
        self,
        config: TkTTSConfig,
        dropout: float = 0.,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        self.dim_model = config.dim_head * config.num_heads  # 模型总维度
        dim_audio = config.fft_length * 3 // 2 + 3  # 输入/输出维度

        # 将 token 映射为向量
        self.embedding = nn.Embedding(config.vocab_size, self.dim_model, device=device)

        # 将组合谱图映射为向量
        self.audio_projection = nn.Linear(dim_audio, self.dim_model)

        # 堆叠多个 TkTTS En/Decoder Layer 层
        layer = TransformerEncoderLayer(config.num_heads, config.dim_head, config.dim_feedforward, dropout=dropout, device=device)
        self.encoder = nn.ModuleList(copy.deepcopy(layer) for _ in range(config.num_layers))
        layer = TransformerDecoderLayer(config.num_heads, config.dim_head, config.dim_feedforward, dropout=dropout, device=device)
        self.decoder = nn.ModuleList(copy.deepcopy(layer) for _ in range(config.num_layers))

        # 输出组合谱图和结束标志
        self.audio_predictor = nn.Linear(self.dim_model, dim_audio, device=device)
        self.stop_predictor = nn.Linear(self.dim_model, 1, device=device)

    @overload
    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor,
        source_padding_mask: Optional[torch.Tensor],
        target_padding_mask: Optional[torch.Tensor],
        kv_cache: None
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], NetKVCache]:
        ...

    @overload
    def forward(
        self,
        source: None,
        target: torch.Tensor,
        source_padding_mask: None,
        target_padding_mask: None,
        kv_cache: NetKVCache
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], NetKVCache]:
        ...

    def forward(
        self,
        source,
        target,
        source_padding_mask=None,
        target_padding_mask=None,
        kv_cache=None
    ):
        if source is None:
            memory = None
        else:
            # 将 token 转为向量，并乘以 sqrt(dim_model) 进行缩放
            memory = self.embedding(source) * math.sqrt(self.dim_model)

            # 通过编码器
            for layer in self.encoder:
                memory = layer(memory, padding_mask=source_padding_mask)

        # 将组合谱图映射为向量
        target = self.audio_projection(target)

        # 通过解码器
        layers_kv_cache = []
        for layer_idx, layer in enumerate(self.decoder):
            target, layer_kv_cache = layer(
                target,
                memory,
                target_padding_mask=target_padding_mask,
                memory_padding_mask=source_padding_mask,
                kv_cache=kv_cache[layer_idx] if kv_cache else None,
            )
            layers_kv_cache.append(layer_kv_cache)

        # 输出组合频谱和结束标志
        audio_prediction = self.audio_predictor(target)
        stop_prediction = F.sigmoid(self.stop_predictor(target)[..., 0])

        return (audio_prediction, stop_prediction), layers_kv_cache
