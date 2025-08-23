# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
import os
from typing import Optional
from collections.abc import Iterator
import librosa
import orjson
from regex import R
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch import nn, optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler, DataLoader
from matplotlib import pyplot as plt
from utils.checkpoint import TagLabelEncoder, TkTTSMetrics, load_checkpoint_train, save_checkpoint
from utils.constants import DEFAULT_ACCUMULATION_STEPS, DEFAULT_DROPOUT, DEFAULT_DUARTION_WEIGHT, DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY
from utils.dataset import AudioMetadata
from utils.model import FastSpeech2
from utils.tookit import parallel_map

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())


class TkTTSDataset(Dataset):
    """
    文本转语音数据集加载器，用于处理音频-文本配对数据

    Args:
        metadata_files: 元数据文件列表

    Yields:
        - 分词后的文本ID序列
        - 正面标签ID序列
        - 负面标签ID序列
        - 对应的梅尔谱图特征矩阵
        - 音频的音高序列
        - 音频的能量序列
    """

    def __init__(self, metadata_files: list[pathlib.Path]):
        self.data_samples = []

        # 获取所有元数据和音频并加入数据列表
        loaded_chunks = {}
        for working_dir, audio_metadata in tqdm([
            (metadata_file.parent, audio_metadata)
            for metadata_file in metadata_files
            for audio_metadata in orjson.loads(metadata_file.read_bytes())
        ]):
            # 加载音频内容分块
            chunk_file = audio_metadata["filename"]
            if chunk_file not in loaded_chunks:
                loaded_chunks[chunk_file] = np.load(working_dir / chunk_file)

            # 添加音频元数据信息和音频到列表
            audio_id = audio_metadata["audio_id"]
            self.data_samples.append((
                audio_metadata["text"],
                audio_metadata["positive_prompt"],
                audio_metadata["negative_prompt"],
                loaded_chunks[chunk_file][f"{audio_id}:mel"],
                loaded_chunks[chunk_file][f"{audio_id}:pitch"],
                loaded_chunks[chunk_file][f"{audio_id}:energy"]
            ))

    def __getitem__(self, index: int) -> tuple[list[int], list[int], list[int], np.ndarray, np.ndarray, np.ndarray]:
        return self.data_samples[index]

    def __len__(self) -> int:
        return len(self.data_samples)


class TkTTSDatasetSampler(Sampler[list[int]]):
    """
    用于 TkTTS 数据集的分批采样器，根据序列长度进行动态批处理。

    该采样器会:
    1. 根据序列长度对样本进行排序
    2. 动态创建批次，确保每个批次的 token 总数不超过 max_batch_tokens
    3. 每个epoch都会重新打乱数据顺序

    Attributes:
        max_batch_tokens: 单个批次允许的最大 token 数量
        seed: 随机种子
        batches: 当前分配到的批次列表

    Examples:
        >>> dataset = TkTTSDataset([pathlib.Path("data/")], tokenizer)
        >>> sampler = TkTTSDatasetSampler(dataset, max_batch_tokens=4096)
        >>> for batch in sampler:
        ...     print(batch)  # [19, 89, 64]
    """

    def __init__(self, dataset: TkTTSDataset, max_batch_tokens: int, seed: int = 8964):
        super().__init__()
        self.max_batch_tokens = max_batch_tokens
        self.seed = seed
        self.batches = []

        # 预计算所有样本的索引和长度
        # dataset.data_samples 数据结构应为 list[tuple[文本 ID 序列, ..., 音频特征]]
        self.index_and_lengths = [(idx, len(dataset.data_samples[idx][0]) + len(dataset.data_samples[idx][-1])) for idx in range(len(dataset))]
        self.index_to_length = dict(self.index_and_lengths)

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前epoch并重新生成批次

        每个epoch开始时调用，用于:
        1. 根据新epoch重新打乱数据顺序
        2. 重新分配批次
        """
        generator = random.Random(self.seed + epoch)

        # 按长度排序，加入随机因子避免固定排序
        sorted_pairs = sorted(self.index_and_lengths, key=lambda pair: (pair[1], generator.random()))

        # 初始化批次列表
        batches_with_tokens: list[tuple[list[int], int]] = []
        current_batch: list[int] = []

        for idx, seq_len in sorted_pairs:
            token_len = seq_len - 1  # 输入序列比原序列少1

            # 处理超长序列
            if token_len > self.max_batch_tokens:
                batches_with_tokens.append(([idx], token_len))
                continue

            # 计算当前批次加入新样本后的token总数
            estimated_tokens = (len(current_batch) + 1) * token_len
            if estimated_tokens > self.max_batch_tokens:
                # 当前批次中最长序列决定了该批次的token总数
                longest_in_batch = self.index_to_length[current_batch[-1]] - 1
                batch_tokens = longest_in_batch * len(current_batch)
                batches_with_tokens.append((current_batch, batch_tokens))
                current_batch = []

            current_batch.append(idx)

        # 添加最后一个批次
        if current_batch:
            longest_in_batch = self.index_to_length[current_batch[-1]] - 1
            batch_tokens = longest_in_batch * len(current_batch)
            batches_with_tokens.append((current_batch, batch_tokens))

        # 将批次列表反转，使得较大的批次优先用于训练，有助于及早发现和修复 OOM（内存溢出）等问题
        batches_with_tokens.reverse()

        # 分配批次
        self.batches = [batch for batch, _ in batches_with_tokens]

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def sequence_collate_fn(batch: list[tuple[list[int], list[int], list[int], np.ndarray, np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, Optional[list[list[int]]], Optional[list[list[int]]], torch.Tensor, torch.Tensor, torch.Tensor, torch.LongTensor, torch.LongTensor]:
    """
    处理语音合成任务的批次数据，将变长序列填充对齐并转换为PyTorch张量。
    
    该函数主要完成以下工作：
    1. 解压批次数据中的文本序列、正负提示词、梅尔频谱、音高和能量特征
    2. 将提示词列表处理为None（如果为空）
    3. 将所有数值数据转换为PyTorch张量
    4. 计算文本和频谱的实际长度
    5. 使用pad_sequence对变长序列进行批量填充
    6. 返回处理后的张量数据及对应长度信息

    Args:
        batch: 包含多个样本的列表，每个样本是元组形式：
            - 文本序列
            - 正提示词序列
            - 负提示词序列 
            - 梅尔频谱
            - 音高特征
            - 能量特征

    Returns:
        处理后的批次数据元组，包含：
            - 填充后的文本序列
            - 正提示词列表（可能为None）
            - 负提示词列表（可能为None）
            - 填充后的梅尔频谱
            - 填充后的音高特征
            - 填充后的能量特征
            - 文本序列实际长度
            - 频谱序列实际长度

    Examples:
        >>> batch = [([1,2,3], [], [], np.array([...]), ...), ...]
        >>> collated = sequence_collate_fn(batch)
    """
    # 解压批次数据
    text_sequences, positive_prompt, negative_prompt, mel_spectrogram, pitch, energy = zip(*batch)

    # 如果提示为空，则将其置为 None
    positive_prompt = positive_prompt or None
    negative_prompt = negative_prompt or None

    # 定义转换为 PyTorch 张量的函数
    convert_to_tensor = lambda x: [torch.tensor(item) for item in x]

    # 转换为 PyTorch 张量
    text_sequences = convert_to_tensor(text_sequences)
    mel_spectrogram = convert_to_tensor(mel_spectrogram)
    pitch = convert_to_tensor(pitch)
    energy = convert_to_tensor(energy)

    # 计算实际序列长度
    text_lengths = torch.tensor([len(seq) for seq in text_sequences], dtype=int)
    audio_lengths = torch.tensor([auido.size(0) for auido in mel_spectrogram], dtype=int)

    # 定义序列填充函数
    fast_pad_sequence = lambda x: torch.nn.utils.rnn.pad_sequence(x, batch_first=True)

    # 对序列进行填充对齐（批次维度在前）
    padded_text = fast_pad_sequence(text_sequences).to(dtype=int)
    padded_audio = fast_pad_sequence(mel_spectrogram)
    padded_pitch = fast_pad_sequence(pitch)
    padded_energy = fast_pad_sequence(energy)

    # 返回批次数据
    return padded_text, positive_prompt, negative_prompt, padded_audio, padded_pitch, padded_energy, text_lengths, audio_lengths


def fastspeech2_loss(
    audio_pred: torch.Tensor,
    duration_pred: torch.Tensor,
    pitch_pred: torch.Tensor,
    energy_pred: torch.Tensor,
    audio_target: torch.Tensor,
    duration_sum_target: torch.Tensor,
    pitch_target: torch.Tensor,
    energy_target: torch.Tensor,
    duration_weight: float
) -> torch.Tensor:
    """
    计算 FastSpeech2 模型的复合损失函数，包含音频重建、时长、音高和能量损失。

    通过生成掩码来识别有效音频帧，分别计算各分量的L1损失，并对有效区域求平均。
    时长损失使用总时长比较，其他损失使用逐帧掩码计算，最后返回加权和。

    Args:
        audio_pred: 预测的梅尔频谱图，形状为 [batch_size, seq_len, mel_dim]
        duration_pred: 预测的音素时长，形状为 [batch_size, phoneme_len]
        pitch_pred: 预测的音高特征，形状为 [batch_size, seq_len]
        energy_pred: 预测的能量特征，形状为 [batch_size, seq_len]
        audio_target: 真实的梅尔频谱图，形状为 [batch_size, seq_len, mel_dim]
        duration_sum_target: 真实的总时长，形状为 [batch_size]
        pitch_target: 真实的音高特征，形状为 [batch_size, seq_len]
        energy_target: 真实的能量特征，形状为 [batch_size, seq_len]
        duration_weight: 持续时间损失的权重系数

    Returns:
        组合损失值，标量张量
    """
    def masked_l1_loss(pred: torch.Tensor, target: torch.Tensor, padding_mask: torch.BoolTensor):
        "计算掩码区域的L1损失，仅对有效帧求平均"
        # 计算逐元素L1损失 [batch_size, seq_len, ...]
        elementwise_loss = F.l1_loss(pred, target, reduction="none")

        # 重塑损失张量以便统一 duration/pitch/energy 和 mel 掩码应用逻辑
        # [batch_size, seq_len, feature_dim]，其中 feature_dim 可能为 1 或 dim_model
        loss_reshaped = elementwise_loss.view(*elementwise_loss.shape[:2], -1)

        # 扩展掩码以匹配损失张量的形状
        expanded_mask = padding_mask.unsqueeze(2).expand_as(loss_reshaped)

        # 将填充区域的损失置零
        maksed_loss = loss_reshaped.masked_fill(expanded_mask, 0)

        # 计算损失平均值
        return maksed_loss.sum() / (~expanded_mask).sum()

    # 标记超出有效时长的填充帧 [batch_size, max_frames]
    frame_indices = torch.arange(audio_pred.size(1), device=audio_pred.device).unsqueeze(0)  # [1, max_frames]
    audio_padding_mask = frame_indices >= duration_pred.sum(dim=1).ceil().unsqueeze(1)  # [batch_size, max_frames]

    # 计算总时长损失
    # 这里不用 padding_mask 是因为，前向传播时已经把填充部分置零了，
    # 所以 duration_pred.sum(dim=1) 已经是去除了填充部分的总和
    duration_loss = F.l1_loss(duration_pred.sum(dim=1), duration_sum_target) * duration_weight

    # 计算各分量损失
    audio_loss = masked_l1_loss(audio_pred, audio_target, audio_padding_mask)
    pitch_loss = masked_l1_loss(pitch_pred, pitch_target, audio_padding_mask)
    energy_loss = masked_l1_loss(energy_pred, energy_target, audio_padding_mask)

    # debug info
    print(f"\n[debug] d={duration_loss.item():.2f}, a={audio_loss.item():.2f}, p={pitch_loss.item():.2f}, e={energy_loss.item():.2f}")
    print(f"d_pred: {duration_pred}")
    print(f"d_sum: {duration_pred.sum(dim=1)}")

    # 返回组合损失
    return audio_loss + duration_loss + pitch_loss + energy_loss


def train(
    model: FastSpeech2,
    dataloader: DataLoader,
    optimizer: optim.AdamW,
    scaler: GradScaler,
    duration_weight: float,
    accumulation_steps: int = 1,
    device: torch.device = torch.device("cpu")
) -> list[float]:
    """
    训练 TkTTS-FastSpeech2 模型的函数，支持梯度累积和混合精度训练

    函数执行流程：
    1. 设置模型为训练模式，初始化梯度缩放器和训练监控工具
    2. 遍历数据加载器中的每个批次
    3. 生成文本和组合谱图的 padding mask
    4. 在混合精度环境下计算模型输出和损失
    5. 执行梯度反向传播和参数更新（根据累积步数）
    6. 记录并返回训练过程中的损失值

    Args:
        model: 待训练的 TkTTS 模型实例
        dataloader: 训练数据加载器
        optimizer: 模型优化器
        scaler: 混合精度梯度缩放器
        accumulation_steps: 梯度累积步数
        duration_weight: 持续时间损失的权重系数
        device: 训练设备（默认使用 CPU）

    Returns:
        每个梯度更新步骤的平均损失值列表

    Examples:
        >>> losses = train(model, loader, opt)
        >>> plt.plot(losses)  # 绘制损失曲线
    """
    # 设置模型为训练模式
    model.train()

    # 初始化损失列表
    losses = []

    # 计算训练多少次迭代
    num_iterators = len(dataloader) // accumulation_steps * accumulation_steps

    # 创建进度条，显示训练进度
    progress_bar = tqdm(total=num_iterators)

    # 当前累积步骤的总损失
    accumulated_loss = 0

    # 提前清空梯度
    optimizer.zero_grad()

    # 遍历整个训练集
    for step, batch in zip(range(num_iterators), dataloader):
        # 数据移至目标设备
        padded_text, positive_prompt, negative_prompt, padded_audio, padded_pitch, padded_energy, text_length, audio_length = [(item.to(device=device) if isinstance(item, torch.Tensor) else item) for item in batch]

        # 准备文本掩码
        text_padding_mask = torch.arange(padded_text.size(1), device=padded_text.device).unsqueeze(0) >= text_length.unsqueeze(1)

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            audio_pred, duration_pred, pitch_pred, energy_pred = model(padded_text, positive_prompt, negative_prompt, text_padding_mask, audio_length, padded_pitch, padded_energy)  # 模型前向传播（使用教师强制）
            loss = fastspeech2_loss(audio_pred, duration_pred, pitch_pred, energy_pred, padded_audio, audio_length.to(dtype=duration_pred.dtype), padded_pitch, padded_energy, duration_weight)  # 计算损失

        # 梯度缩放与反向传播
        scaler.scale(loss).backward()
        accumulated_loss += loss.item()

        # 达到累积步数时更新参数
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 调整缩放因子
            optimizer.zero_grad()  # 清空梯度

            accumulated_loss /= accumulation_steps
            losses.append(accumulated_loss)  # 记录平均损失
            progress_bar.set_postfix(loss=accumulated_loss)  # 更新进度条
            accumulated_loss = 0  # 重置累积损失

        # 更新进度条
        progress_bar.update()

    return losses


@torch.inference_mode
def validate(
    model: FastSpeech2,
    dataloader: DataLoader,
    duration_weight: float,
    device: torch.device = torch.device("cpu")
) -> list[float]:
    """
    在验证集上评估 FastSpeech2 模型的性能
    计算模型在验证集上的平均损失值，用于监控训练过程和模型选择
    整个过程使用推理模式，禁用梯度计算和自动微分以节省内存和加速计算

    Args:
        model: 要验证的FastSpeech2模型实例
        dataloader: 验证集的数据加载器
        duration_weight: 持续时间损失的权重系数
        device: 计算设备，默认为CPU

    Returns:
        包含每个批次损失值的列表

    Examples:
        >>> losses = validate(model, val_loader, 1.0, torch.device("cuda"))
        >>> avg_loss = sum(losses) / len(losses)
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化损失列表
    losses = []

    # 遍历整个验证集
    for batch in tqdm(dataloader, total=len(dataloader)):
        # 数据移至目标设备
        padded_text, positive_prompt, negative_prompt, padded_audio, padded_pitch, padded_energy, text_length, audio_length = [(item.to(device=device) if isinstance(item, torch.Tensor) else item) for item in batch]

        # 准备文本掩码
        text_padding_mask = torch.arange(padded_text.size(1), device=padded_text.device).unsqueeze(0) >= text_length.unsqueeze(1)

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            # 模型前向传播（不使用教师强制）
            audio_pred, duration_pred, pitch_pred, energy_pred = model(padded_text, positive_prompt, negative_prompt, text_padding_mask)

            # 填充序列，方便计算损失
            batch_size, audio_pred_len, num_mels = audio_pred.size()
            audio_target_len = padded_audio.size(1)
            max_len = max(audio_pred_len, audio_target_len)
            audio_pred = torch.cat([audio_pred, torch.zeros(batch_size, max_len - audio_pred_len, num_mels, device=audio_pred.device)], dim=1)
            pitch_pred = torch.cat([pitch_pred, torch.zeros(batch_size, max_len - audio_pred_len, device=audio_pred.device)], dim=1)
            energy_pred = torch.cat([energy_pred, torch.zeros(batch_size, max_len - audio_pred_len, device=audio_pred.device)], dim=1)
            padded_audio = torch.cat([padded_audio, torch.zeros(batch_size, max_len - audio_target_len, num_mels, device=audio_pred.device)], dim=1)
            padded_pitch = torch.cat([padded_pitch, torch.zeros(batch_size, max_len - audio_target_len, device=audio_pred.device)], dim=1)
            padded_energy = torch.cat([padded_energy, torch.zeros(batch_size, max_len - audio_target_len, device=audio_pred.device)], dim=1)

            # 计算损失
            loss = fastspeech2_loss(audio_pred, duration_pred, pitch_pred, energy_pred, padded_audio, audio_length.to(dtype=duration_pred.dtype), padded_pitch, padded_energy, duration_weight)

        # 记录损失
        losses.append(loss.item())

    return losses


def plot_training_process(metrics: TkTTSMetrics, img_path: pathlib.Path | str, show_training_process: bool = False):
    """
    绘制损失变化过程。训练损失使用红线，验证损失用蓝色点线。
    为每种损失分别绘制置信区间。

    Args:
        metrics: 训练历史记录
        img_path: 图形保存的文件路径，可以是字符串或Path对象。
        show_training_process: 展示训练过程图表，而不仅仅是保存为文件

    Example:
        ```
        metrics = {
            "train_loss": [
                {"mean": 1.2, "std": 0.1, "count": 100},
                {"mean": 1.0, "std": 0.08, "count": 100},
            ],
            "val_loss": [
                {"mean": 1.1, "std": 0.09},
                {"mean": 0.95, "std": 0.07},
            ]
        }
        ```
    """
    # 创建图形和坐标轴
    _, ax = plt.subplots(figsize=(10, 6))

    # 计算验证点的x坐标（每个epoch的起始位置）
    current_iteration = metrics["train_loss"][0]["count"]  # 当前累计的迭代次数
    val_iteration_points = [current_iteration]  # 存储每个epoch的起始迭代次数
    for epoch in metrics["train_loss"][1:]:
        current_iteration += epoch["count"]  # 累加当前epoch的迭代次数
        val_iteration_points.append(current_iteration)

    # 计算训练损失曲线的x坐标（偏移半个epoch）
    # 这里将每个训练损失点放在其对应 epoch 区间的中间位置（即当前验证点左移半个 epoch），
    # 这样可以更直观地反映该 epoch 内的训练损失均值与验证损失的时间关系。
    train_x = [val_iteration_points[i] - epoch["count"] / 2 for i, epoch in enumerate(metrics["train_loss"])]

    # 绘制训练损失曲线和标准差区间
    ax.plot(train_x, [epoch["mean"] for epoch in metrics["train_loss"]], label="Train Loss", color="red", linestyle="-", marker=".")
    train_upper = [epoch["mean"] + epoch["std"] for epoch in metrics["train_loss"]]
    train_lower = [epoch["mean"] - epoch["std"] for epoch in metrics["train_loss"]]
    ax.fill_between(train_x, train_upper, train_lower, color="red", alpha=0.2)

    ax.plot(val_iteration_points, [epoch["mean"] for epoch in metrics["val_loss"]], label="Validation Loss", color="blue", linestyle="-", marker=".")
    val_upper = [epoch["mean"] + epoch["std"] for epoch in metrics["val_loss"]]
    val_lower = [epoch["mean"] - epoch["std"] for epoch in metrics["val_loss"]]
    ax.fill_between(val_iteration_points, val_upper, val_lower, color="blue", alpha=0.2)

    # 设置X轴为整数刻度
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))

    # 设置坐标轴标签和标题
    ax.set_ylabel("Loss")
    ax.set_xlabel("Iteration")
    ax.set_title("Training Process")

    # 添加图例和网格
    ax.legend(loc="upper right")
    ax.grid(True, linestyle="--", alpha=0.5)

    # 保存图形
    plt.tight_layout()
    pathlib.Path(img_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_path, dpi=300, bbox_inches="tight")

    # 展示图形
    if show_training_process:
        plt.show()

    plt.close()


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="训练 TkTTS 模型")
    parser.add_argument("num_epochs", type=int, help="训练的总轮数")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="加载和保存检查点的路径")
    parser.add_argument("-t", "--train-dataset", action="append", type=pathlib.Path, required=True, help="训练集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-v", "--val-dataset", action="append", type=pathlib.Path, help="验证集文件路径（可多次指定以使用多个数据集）")
    parser.add_argument("-tt", "--train-max-batch-tokens", default=4096, type=int, help="训练时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-tv", "--val-max-batch-tokens", default=16384, type=int, help="验证时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-lr", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率，默认为 %(default)s")
    parser.add_argument("-wd", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数，默认为 %(default)s")
    parser.add_argument("-do", "--dropout", default=DEFAULT_DROPOUT, type=float, help="Dropout 概率，用于防止过拟合，默认为 %(default)s")
    parser.add_argument("-as", "--accumulation-steps", default=DEFAULT_ACCUMULATION_STEPS, type=int, help="梯度累积步数，默认为 %(default)s")
    parser.add_argument("-dw", "--duration-weight", default=DEFAULT_DUARTION_WEIGHT, type=float, help="持续时间损失的权重系数")
    parser.add_argument("-k", "--show-training-process", action="store_true", help="展示训练过程图表，而不仅仅是保存到`<ckpt_path>/statistics.png`")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 设置当前进程的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取检查点
    model_state_dict, model_config, optimiezer_state_dict, metrics = load_checkpoint_train(args.ckpt_path)

    # 创建模型并加载状态
    model = FastSpeech2(model_config, dropout=args.dropout)
    model.load_state_dict(model_state_dict)

    # 转移模型到设备
    model = model.to(device)

    # 多 GPU 时使用 DataParallel 包装模型
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    # 创建优化器并加载状态
    optimizer = optim.AdamW(model.parameters())
    optimizer.load_state_dict(optimiezer_state_dict)

    # 设置学习率、权重衰减系数
    for group in optimizer.param_groups:
        group["lr"] = args.learning_rate
        group["weight_decay"] = args.weight_decay

    # 创建混合精度梯度缩放器
    scaler = GradScaler(device)

    # 加载训练数据集
    train_dataset = TkTTSDataset(args.train_dataset)
    train_sampler = TkTTSDatasetSampler(train_dataset, args.train_max_batch_tokens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 如果存在验证集，加载验证数据集
    if args.val_dataset:
        val_dataset = TkTTSDataset(args.val_dataset)
        val_sampler = TkTTSDatasetSampler(val_dataset, args.val_max_batch_tokens)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 开始训练
    for epoch in range(args.num_epochs):
        # 计算累积 Epoch 数
        current_epoch = len(metrics["train_loss"]) + epoch

        # 训练一轮模型
        train_sampler.set_epoch(current_epoch)
        train_loss = train(model, train_loader, optimizer, scaler, args.duration_weight, accumulation_steps=args.accumulation_steps, device=device)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失为 NaN
        if args.val_dataset:
            val_sampler.set_epoch(current_epoch)
            val_loss = validate(model, val_loader, args.duration_weight, device)
        else:
            val_loss = [float("nan")]

        # 计算并添加损失平均值和标准差到指标
        train_loss = np.array(train_loss)
        val_loss = np.array(val_loss)
        metrics["train_loss"].append({"mean": train_loss.mean().item(), "std": train_loss.std().item(), "count": len(train_loss)})
        metrics["val_loss"].append({"mean": val_loss.mean().item(), "std": val_loss.std().item()})

    # 保存当前模型的检查点
    save_checkpoint(
        args.ckpt_path,
        (model.module if torch.cuda.device_count() > 1 else model).cpu().state_dict(),
        optimizer.state_dict(),
        metrics,
    )
    print(f"训练完成，模型已保存到 {args.ckpt_path}。")

    # 绘制训练过程中的损失曲线
    plot_training_process(metrics, args.ckpt_path / "statistics.png", show_training_process=args.show_training_process)


if __name__ == "__main__":
    main(parse_args())
