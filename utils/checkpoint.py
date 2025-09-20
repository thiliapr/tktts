# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
from typing import Any, Optional, TypedDict
from collections.abc import Mapping
import orjson
import torch
from utils.model import FastSpeech2Config


class TagLabelEncoder:
    """
    标签编码器，用于将标签字符串转换为整数ID。
    主要用于处理模型的标签输入和输出。
    PAD 标签必须存在且 ID 为 0。

    Args:
        tags: 标签字典，键为标签字符串，值为对应的整数ID。
    
    Attributes:
        vocab: 标签字典，包含所有标签及其对应的ID。
        id_to_tag: 反向映射字典，将ID映射回标签字符串。
        pad_token: 填充标签
    """

    def __init__(self, tags: dict[str, int]):
        # 初始化标签编码器
        self.vocab = tags
        self.id_to_tag = {v: k for k, v in tags.items()}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, tag_list: list[str]) -> list[int]:
        """
        将标签列表编码为整数ID列表。
        
        Args:
            tag_list: 标签字符串列表。
        
        Returns:
            包含标签对应整数ID的列表。如果标签不在词汇表中，则忽略该标签。
        
        Examples:
            >>> encoder = TagLabelEncoder({"character:白上フブキ": 0, "character:夏色まつり": 1})
            >>> encoder.encode(["character:白上フブキ", "character:夏色まつり", "unknown_tag"])
            [0, 1]  # unknown_tag 不在词汇表中，因此被忽略
        """
        return [self.vocab[tag] for tag in tag_list if tag in self.vocab]


class TkTTSExtraConfig(TypedDict):
    """
    TkTTS-FreeSpeech2 额外的配置。
    具体来说，是无法从模型状态推断的配置，比如训练模型时音频的预处理配置等。

    Attributes:
        num_heads: 注意力头的数量
        sample_rate: 目标采样率
        fft_length: FFT 窗口长度
        frame_length: YIN 算法的分析帧长度
        hop_length: 帧移长度
        win_length: 窗函数长度
    """
    num_heads: int
    sample_rate: int
    fft_length: int
    frame_length: int
    hop_length: int
    win_length: int


def save_checkpoint(
    path: pathlib.Path,
    model_state: Mapping[str, Any],
    optimizer_state: Mapping[str, Any],
    completed_epochs: int,
    extra_config: Optional[TkTTSExtraConfig] = None,
    phoneme_encoder: Optional[TagLabelEncoder] = None,
    tag_label_encoder: Optional[TagLabelEncoder] = None
):
    """
    保存模型的检查点到指定路径，包括模型的权重以及训练的进度信息。

    Args:
        path: 保存检查点的目录路径
        model_state: 要保存的模型的状态字典
        optimizer_state: 要保存的优化器的状态字典
        completed_epochs: 总共训练了多少个 Epoch
        extra_config: 模型额外配置（用于预处理数据、推理生成）
        phoneme_encoder: 音素编码器，用于处理音素的编码和解码
        tag_label_encoder: 标签编码器，用于处理标签的编码和解码
    """
    path.mkdir(parents=True, exist_ok=True)  # 确保目标目录存在，如果不存在则创建
    torch.save(model_state, path / "model.pth")  # 保存模型权重
    torch.save(optimizer_state, path / "optimizer.pth")  # 保存优化器权重

    # 保存进度信息
    (path / "progress.json").write_bytes(orjson.dumps({"completed_epochs": completed_epochs}))

    # 保存模型配置
    if extra_config:
        (path / "extra_config.json").write_bytes(orjson.dumps(extra_config))

    # 保存音素编码器
    if phoneme_encoder:
        (path / "phoneme.json").write_bytes(orjson.dumps(phoneme_encoder.vocab))

    # 保存标签编码器
    if tag_label_encoder:
        (path / "tags.json").write_bytes(orjson.dumps(tag_label_encoder.vocab))


def load_checkpoint(path: pathlib.Path) -> tuple[TagLabelEncoder, Mapping[str, Any], TkTTSExtraConfig, FastSpeech2Config, TagLabelEncoder]:
    """
    从指定路径加载模型的检查点（用于推理）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        音素编码器、模型的状态、模型额外的配置、用于创建模型的配置、标签编码器

    Examples:
        >>> phoneme_encoder, sd, extra_config, model_config, tag_label_encoder = load_checkpoint(pathlib.Path("ckpt"))
        >>> model = MidiNet(model_config, device=torch.device("cuda"))
        >>> model.load_state_dict(sd)
    """
    # 加载音素编码器
    phoneme_encoder = TagLabelEncoder(orjson.loads((path / "phoneme.json").read_bytes()))

    # 加载模型权重
    model_state = torch.load(path / "model.pth", weights_only=True, map_location=torch.device("cpu"))

    # 加载模型生成配置
    extra_config = orjson.loads((path / "extra_config.json").read_bytes())

    # 提取模型配置
    model_config = extract_config(model_state, extra_config["num_heads"])

    # 加载标签编码器
    tag_label_encoder = TagLabelEncoder(orjson.loads((path / "tags.json").read_bytes()))
    return phoneme_encoder, model_state, extra_config, model_config, tag_label_encoder


def load_checkpoint_train(path: pathlib.Path) -> tuple[Mapping[str, Any], TkTTSExtraConfig, FastSpeech2Config, TagLabelEncoder, Mapping[str, Any], int]:
    """
    从指定路径加载模型的检查点（用于恢复训练状态）。

    Args:
        path: 加载检查点的目录路径

    Returns:
        模型的状态、模型额外的配置、用于创建模型的配置、标签编码器、优化器的状态、训练了多少个 Epoch

    Examples:
        >>> msd, extra_config, model_config, tag_label_encoder, osd, completed_epochs = load_checkpoint_train(pathlib.Path("ckpt"))
        >>> model = MidiNet(model_config, deivce=torch.device("cuda"))
        >>> model.load_state_dict(msd)
        >>> optimizer = optim.AdamW(model.parameters())
        >>> optimizer.load_state_dict(osd)
        >>> # 继续训练
    """
    # 加载音素编码器和模型状态
    _, model_state, extra_config, model_config, tag_label_encoder = load_checkpoint(path)

    # 检查并加载优化器权重
    optimizer_state = torch.load(path / "optimizer.pth", weights_only=True, map_location=torch.device("cpu"))

    # 读取训练了多少个 Epoch
    completed_epochs = orjson.loads((path / "progress.json").read_bytes())["completed_epochs"]

    # 返回训练所需信息
    return model_state, extra_config, model_config, tag_label_encoder, optimizer_state, completed_epochs


def extract_config(model_state: dict[str, Any], num_heads: int) -> FastSpeech2Config:
    """
    从模型状态字典中提取 TkTTS-FastSpeech2 模型的配置参数
    通过分析 state_dict 中各层的维度大小和结构，自动推断出模型的超参数配置

    Args:
        model_state: 保存模型参数的状态字典
        dim_model: 注意力头数量。如果模型维度不能被注意力头数整除，则会抛出 AssertError

    Returns:
        包含所有提取出的配置参数的 FastSpeech2Config 对象

    Examples:
        >>> state_dict = torch.load("model.pth")
        >>> generation_config = json.loads(pathlib.Path("config.json").read_text("utf-8"))
        >>> config = extract_config(state_dict, generation_config["num_heads"])
    """
    vocab_size, dim_model = model_state["embedding.weight"].size()
    num_tags = model_state["tag_embedding.weight"].size(0)
    num_mels = model_state["mel_predictor.weight"].size(0)
    assert dim_model % num_heads == 0, f"模型维度应该能被注意力头数整除，但并没有: 模型维度({dim_model}) % 注意力头数({num_heads}) != 0"
    num_heads, dim_head = num_heads, dim_model // num_heads
    dim_feedforward, _, fft_conv1_kernel_size = model_state["encoder.0.conv1.model.weight"].size()
    fft_conv2_kernel_size = model_state["encoder.0.conv2.model.weight"].size(2)
    predictor_kernel_size = model_state["pitch_predictor.conv1.model.weight"].size(2)
    variance_bins = model_state["pitch_embedding.weight"].size(0)
    _, postnet_hidden_dim, postnet_kernel_size = model_state["postnet.output_conv.model.weight"].size()
    num_encoder_layers = len({int(key.split(".")[1]) for key in (key for key in model_state if key.startswith("encoder."))})
    num_decoder_layers = len({int(key.split(".")[1]) for key in (key for key in model_state if key.startswith("decoder."))})
    num_postnet_layers = len({int(key.split(".")[2]) for key in (key for key in model_state if key.startswith("postnet.layers."))}) + 1
    return FastSpeech2Config(
        vocab_size=vocab_size,
        num_tags=num_tags,
        num_mels=num_mels,
        num_heads=num_heads,
        dim_head=dim_head,
        dim_feedforward=dim_feedforward,
        fft_conv1_kernel_size=fft_conv1_kernel_size,
        fft_conv2_kernel_size=fft_conv2_kernel_size,
        predictor_kernel_size=predictor_kernel_size,
        postnet_hidden_dim=postnet_hidden_dim,
        postnet_kernel_size=postnet_kernel_size,
        variance_bins=variance_bins,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        num_postnet_layers=num_postnet_layers
    )