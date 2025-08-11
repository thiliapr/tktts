# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
from typing import Any, Optional, TypedDict, Union
from collections.abc import Mapping
import orjson
import torch
from transformers import AutoTokenizer
from utils.model import TkTTSConfig


class GenerationConfig(TypedDict):
    """
    TkTTS 生成用的配置。

    Attributes:
        num_heads: 注意力头的数量
        sample_rate: 目标采样率
        hop_length: 帧移长度
        win_length: 窗函数长度
    """
    num_heads: int
    sample_rate: int
    hop_length: int
    win_length: int


class TkTTSMetrics(TypedDict):
    """
    TkTTS 训练历史。

    Attributes:
        train_loss: 训练批次数，损失平均值、标准差
        val_loss: 验证损失平均值、标准差
    """
    train_loss: list[dict[str, Union[float, int]]]
    val_loss: list[dict[str, Union[float, int]]]


def save_checkpoint(
    path: pathlib.Path,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    metrics: TkTTSMetrics,
    generation_config: Optional[GenerationConfig] = None
):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        path: 保存检查点的目录路径
        model_state: 要保存的模型的状态字典
        optimizer_state: 要保存的优化器的状态字典
        metrics: 模型的训练过程（每个Epoch的训练损失和验证损失）
        generation_config: 模型生成配置（用于预处理数据、推理生成）
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建
    torch.save(model_state, path / "model.pth")  # 保存模型权重
    torch.save(optimizer_state, path / "optimizer.pth")  # 保存优化器权重

    # 保存训练信息
    (path / "metrics.json").write_bytes(orjson.dumps(metrics))

    # 保存模型配置
    if generation_config:
        (path / "generation_config.json").write_bytes(orjson.dumps(generation_config))


def load_checkpoint(path: pathlib.Path) -> tuple[AutoTokenizer, Mapping[str, Any], GenerationConfig, TkTTSConfig]:
    """
    从指定路径加载模型的检查点（用于推理）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        分词器、模型的状态、模型生成的配置、用于创建模型的配置

    Examples:
        >>> tokenizer, sd, generation_config, model_config = load_checkpoint(pathlib.Path("ckpt"))
        >>> model = MidiNet(model_config, device=torch.device("cuda"))
        >>> model.load_state_dict(sd)
    """
    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(path / "tokenizer")

    # 加载模型权重
    model_state = torch.load(path / "model.pth", weights_only=True, map_location=torch.device("cpu"))

    # 加载模型生成配置
    generation_config = orjson.loads((path / "generation_config.json").read_bytes())

    # 提取模型配置
    model_config = extract_config(model_state, generation_config["num_heads"])
    return tokenizer, model_state, generation_config, model_config


def load_checkpoint_train(path: pathlib.Path) -> tuple[AutoTokenizer, Mapping[str, Any], GenerationConfig, TkTTSConfig, Mapping[str, Any], TkTTSMetrics]:
    """
    从指定路径加载模型的检查点（用于恢复训练状态）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        分词器、模型的状态、模型生成的配置、用于创建模型的配置、优化器的状态、指标

    Examples:
        >>> tokenizer, msd, model_config, generation_config, osd, metrics = load_checkpoint_train(pathlib.Path("ckpt"))
        >>> model = MidiNet(model_config, deivce=torch.device("cuda"))
        >>> model.load_state_dict(msd)
        >>> optimizer = optim.AdamW(model.parameters())
        >>> optimizer.load_state_dict(osd)
        >>> # 继续训练...
    """
    # 加载分词器和模型状态
    tokenizer, model_state, generation_config, model_config = load_checkpoint(path)

    # 检查并加载优化器权重
    optimizer_state = torch.load(path / "optimizer.pth", weights_only=True, map_location=torch.device("cpu"))

    # 尝试加载指标文件
    metrics = orjson.loads((path / "metrics.json").read_bytes())
    return tokenizer, model_state, generation_config, model_config, optimizer_state, metrics


def extract_config(model_state: dict[str, Any], num_heads: int) -> TkTTSConfig:
    """
    从模型状态字典中提取 TkTTS 模型的配置参数
    通过分析 state_dict 中各层的维度大小和结构，自动推断出模型的超参数配置

    工作流程：
    1. 从 embedding 层获取词汇表大小和模型维度
    2. 从 feedforward 层获取前馈网络维度
    3. 统计 encoder 层数量
    4. 从 spec_proj 层获取 fft_length
    5. 根据模型维度自动计算注意力头数和每个头的维度

    Args:
        model_state: 保存模型参数的状态字典
        dim_model: 注意力头数量。如果模型维度不能被注意力头数整除，则会抛出 AssertError

    Returns:
        包含所有提取出的配置参数的TkTTSConfig对象

    Examples:
        >>> state_dict = torch.load("model.pth")
        >>> generation_config = json.loads(pathlib.Path("config.json").read_text("utf-8"))
        >>> config = extract_config(state_dict, generation_config["num_heads"])
    """
    # 从 embedding 层获取词汇表大小和模型维度
    vocab_size, dim_model = model_state["embedding.weight"].size()

    # 从第一个 encoder 层的 feedforward 层获取前馈网络维度
    dim_feedforward = model_state["encoder.0.feedforward.0.weight"].size(0)

    # 统计 encoder 层数量（通过统计不同层编号）
    encoder_layer_keys = (key for key in model_state if key.startswith("encoder."))
    layer_indices = {int(key.split(".")[1]) for key in encoder_layer_keys}
    num_layers = len(layer_indices)

    # 从输出层获取 FFT 窗口长度
    fft_length = model_state["audio_predictor.weight"].size(0) * 2 // 3 - 2

    # 自动计算注意力头数和每个头的维度
    assert dim_model % num_heads == 0, f"模型维度应该能被注意力头数整除，但并没有: 模型维度({dim_model}) % 注意力头数({num_heads}) != 0"
    num_heads, dim_head = num_heads, dim_model // num_heads

    return TkTTSConfig(
        vocab_size=vocab_size,
        fft_length=fft_length,
        num_heads=num_heads,
        dim_head=dim_head,
        dim_feedforward=dim_feedforward,
        num_layers=num_layers
    )
