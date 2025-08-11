# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
from os import cpu_count
from typing import Optional
from collections.abc import Iterator
import librosa
import orjson
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from torch import nn, optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler, DataLoader
from matplotlib import pyplot as plt
from utils.checkpoint import TkTTSMetrics, load_checkpoint_train, save_checkpoint
from utils.constants import DEFAULT_DROPOUT, DEFAULT_LEARNING_RATE, DEFAULT_WEIGHT_DECAY
from utils.model import TkTTS
from utils.tookit import parallel_map


class TkTTSDataset(Dataset):
    """
    文本转语音数据集加载器，用于处理音频-文本配对数据

    工作流程：
    1. 扫描指定目录中的所有JSON元数据文件
    2. 使用多进程并行加载和预处理数据：
       - 加载文本元数据并进行分词编码
       - 加载对应音频文件并计算有效帧数
    3. 在访问数据时实时生成声学特征：
       - 加载音频并去除静音段
       - 计算幅度谱和相位谱
       - 拼接组合特征矩阵

    Args:
        data_dirs: 包含JSON元数据文件的根目录列表
        tokenizer: 用于文本编码的预训练分词器
        num_workers: 并行加载数据的工作进程数（默认为CPU核心数）
        sample_rate: 音频采样率（Hz）
        fft_length: FFT窗口长度
        hop_length: 帧移长度
        win_length: 窗函数长度

    Yields:
        - 分词后的文本ID序列
        - 对应的组合谱图特征矩阵

    Examples:
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        >>> dataset = TkTTSDataset([Path("/data/tts1"), Path("/data/tts2")], tokenizer)
        >>> dataloader = DataLoader(dataset, batch_size=32)
        >>> for batch in dataloader:
        >>>     inputs, specs = batch
    """

    def __init__(
        self,
        data_dirs: list[pathlib.Path],
        tokenizer: AutoTokenizer,
        sample_rate: float,
        fft_length: int,
        hop_length: int,
        win_length: int,
        num_workers: int = cpu_count()
    ):
        self.sample_rate = sample_rate
        self.fft_length = fft_length
        self.hop_length = hop_length
        self.win_length = win_length

        # 递归扫描所有目录中的JSON元数据文件
        metadata_files = [
            file
            for data_dir in data_dirs
            for file in data_dir.rglob("*.*")
            if file.suffix.lower() == ".json"
        ]

        # 将文件均匀分配到各工作进程
        worker_results = parallel_map(self._load_worker_data, [
            (worker_id, metadata_files[worker_id::num_workers], tokenizer, sample_rate, hop_length)
            for worker_id in range(num_workers)
        ])

        # 合并所有工作进程的结果
        self.data_samples = [item for worker_data in worker_results for item in worker_data]

    @staticmethod
    def _load_worker_data(
        rank: int,
        metadata_files: list[pathlib.Path],
        tokenizer: AutoTokenizer,
        sample_rate: float,
        hop_length: int,
    ) -> list[tuple[pathlib.Path, list[int], int]]:
        """
        工作进程数据处理函数，加载编码文本并预测音频帧数

        Args:
            worker_id: 工作进程ID（用于进度条控制）
            metadata_files: 分配给当前工作进程的文件列表
            tokenizer: 文本编码分词器
            sample_rate: 音频采样率（Hz）
            hop_length: 帧移长度

        Returns:
            包含(音频文件, 文本token, 音频帧数)的元组列表
        """
        worker_data = []

        for metadata_file in tqdm(metadata_files, disable=rank != 0):
            # 加载文本元数据
            metadata = orjson.loads(metadata_file.read_bytes())

            # 组合标签和文本，使用分词器编码
            tags_section = "".join(f"[{tag}]" for tag in metadata["tags"])
            full_text = tags_section + metadata["text"]
            text_sequences = tokenizer.encode(full_text)

            # 获取对应的音频文件路径（去除.json扩展名）
            audio_path = metadata_file.with_suffix("")

            # 检查音频文件是否存在
            if not audio_path.exists():
                continue

            # 加载音频并重采样
            audio, _ = librosa.load(audio_path, sr=sample_rate)

            # 去除首尾静音段
            audio, _ = librosa.effects.trim(audio)

            # 计算特征帧数
            num_frames = (len(audio) + hop_length - 1) // hop_length
            worker_data.append((audio_path, text_sequences, num_frames))

        return worker_data

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        audio_path, text_sequences, _ = self.data_samples[index]

        # 加载音频并重采样
        audio, _ = librosa.load(audio_path, sr=self.sample_rate)

        # 去除首尾静音段
        audio, _ = librosa.effects.trim(audio)

        # 计算短时傅里叶变换
        linear_spectrogram = librosa.stft(audio, n_fft=self.fft_length, hop_length=self.hop_length, win_length=self.win_length)

        # 提取幅度和相位
        magnitude = np.abs(linear_spectrogram)  # 幅度谱 (n_fft // 2 + 1, T)
        phase = np.angle(linear_spectrogram)  # 相位谱 (n_fft // 2 + 1, T)

        # 使用 cos 和 sin 避免相位跳跃
        phase_cos = np.cos(phase)
        phase_sin = np.sin(phase)

        # 组合声学特征
        spectrogram = np.concatenate([
            magnitude,
            phase_cos,
            phase_sin
        ], axis=0)  # (n_fft // 2 * 3 + 3, T)

        # 添加起始帧
        spectrogram = np.concatenate([np.zeros((spectrogram.shape[0], 1)), spectrogram], axis=1)

        # 转置为 [时间帧, 特征] 并转换类型
        spectrogram = spectrogram.T.astype(np.float32)
        return text_sequences, spectrogram

    def __len__(self) -> int:
        return len(self.data_samples)


class TkTTSDatasetSampler(Sampler[list[int]]):
    """
    用于 TkTTS 数据集的分批采样器，根据序列长度进行动态批处理。

    该采样器会:
    1. 根据序列长度对样本进行排序
    2. 动态创建批次，确保每个批次的token总数不超过max_batch_tokens
    3. 每个epoch都会重新打乱数据顺序

    Attributes:
        max_batch_tokens: 单个批次允许的最大token数量
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
        self.index_and_lengths = [(idx, len(dataset.data_samples[idx][1]) + dataset.data_samples[idx][2]) for idx in range(len(dataset))]
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


def sequence_collate_fn(batch: list[tuple[np.ndarray, np.ndarray]]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    处理变长序列数据的批处理函数
    将变长的文本序列和组合谱图批量处理为填充后的张量，并返回序列实际长度

    工作流程：
    1. 解压批次数据，分离文本序列和组合谱图
    2. 将文本序列转为PyTorch张量
    3. 计算文本序列和组合谱图的实际长度
    4. 对序列进行填充对齐，使批内所有序列等长
    5. 返回填充后的文本序列、组合谱图及各自的实际长度

    Args:
        batch_data: 批次数据列表，每个元素为 (文本序列, 组合谱图) 元组
            - 文本序列: 1D numpy数组，包含token ID序列
            - 组合谱图: 2D numpy数组，形状为 [帧数, 特征维度]

    Returns:
        - 填充后的文本序列张量，形状为 [批次大小, 最大序列长度]
        - 填充后的组合谱图张量，形状为 [批次大小, 最大帧数, 特征维度]
        - 文本序列实际长度张量，形状为 [批次大小]
        - 组合谱图实际长度张量（含起始帧），形状为 [批次大小]

    Examples:
        >>> batch = [
        ...     (np.array([1,2,3]), np.array([[0.1,0.2], [0.3,0.4]])),
        ...     (np.array([4,5]), np.array([[0.5,0.6]]))
        ... ]
        >>> padded_text, padded_spec, text_lens, spec_lens = sequence_collate_function(batch)
    """
    # 解压批次数据: 文本序列、组合谱图
    text_sequences, spectrogram_combineds = zip(*batch)

    # 转换文本序列、组合谱图为PyTorch张量
    text_sequences = [torch.Tensor(text_sequence) for text_sequence in text_sequences]
    spectrogram_combineds = [torch.Tensor(spectrogram_combined) for spectrogram_combined in spectrogram_combineds]

    # 计算实际序列长度
    text_lengths = torch.tensor([len(seq) for seq in text_sequences], dtype=int)
    spec_lengths = torch.tensor([spec.size(0) for spec in spectrogram_combineds], dtype=int)

    # 对序列进行填充对齐（批次维度在前）
    padded_text = torch.nn.utils.rnn.pad_sequence(text_sequences, batch_first=True).to(dtype=int)
    padded_spec = torch.nn.utils.rnn.pad_sequence(spectrogram_combineds, batch_first=True)

    return padded_text, padded_spec, text_lengths, spec_lengths


def prepare_masks_and_targets(
    text_inputs: torch.Tensor,
    spec_targets: torch.Tensor,
    text_lengths: torch.Tensor,
    spec_lengths: torch.Tensor,
    device: torch.device = torch.device("cpu")
) -> tuple[torch.BoolTensor, torch.BoolTensor, torch.Tensor, torch.Tensor]:
    """
    为序列模型训练准备掩码和目标信号

    该函数主要完成以下任务：
    1. 为文本序列和组合谱图序列生成填充位置掩码
    2. 创建停止目标信号（用于预测序列结束位置）
    3. 计算停止目标的权重（特别加强序列末尾的权重）

    Args:
        text_inputs: 文本序列张量，形状为 (batch_size, max_text_len)
        spec_targets: 组合谱图图张量，形状为 (batch_size, max_spec_len, fft_length * 3 // 2 + 3)
        text_lengths: 每个样本的实际文本长度，形状为 (batch_size,)
        spec_lengths: 每个样本的实际音频帧长度，形状为 (batch_size,)
        device: 计算设备（默认CPU）

    Returns:
        - 文本填充掩码，True表示填充位置，形状 (batch_size, max_text_len)
        - 组合谱图填充掩码，True表示填充位置，形状 (batch_size, max_spec_len)
        - 停止目标信号，形状 (batch_size, max_spec_len - 1)
        - 停止目标权重，形状 (batch_size, max_spec_len - 1)

    Examples:
        >>> text = torch.randint(0, 10, (2, 5))
        >>> specs = torch.randn(2, 10, 3075)
        >>> text_lens = torch.tensor([3, 5])
        >>> spec_lens = torch.tensor([8, 6])
        >>> t_mask, m_mask, stop_t, stop_w = prepare_masks_and_targets(text, specs, text_lens, spec_lens)
    """
    # 生成文本序列的填充掩码
    # 比较位置索引与有效长度，生成布尔掩码（True表示填充位置）
    text_mask = torch.arange(text_inputs.size(1), device=device) >= text_lengths.unsqueeze(1)

    # 生成组合谱图的填充掩码
    spec_mask = torch.arange(spec_targets.size(1), device=device) >= spec_lengths.unsqueeze(1)

    # 创建停止目标信号（右移一个位置的spec_mask）
    # 使用spec_mask[1:]位置作为停止信号（1表示需要停止的位置）
    stop_targets = spec_mask[:, 1:].to(dtype=spec_targets.dtype)

    # 计算停止目标权重
    # 1. 基础权重：非填充位置为1，填充位置为0
    stop_weights = (~spec_mask[:, 1:]).to(dtype=int)

    # 2. 加强序列末尾的权重：将每个样本倒数第二个位置的权重设为序列长度-2
    # 使用长度-2是因为序列已右移一位（排除第一个时间步）
    batch_indices = torch.arange(len(spec_lengths), dtype=int, device=device)
    stop_weights[batch_indices, spec_lengths - 2] = spec_lengths - 2

    return text_mask, spec_mask, stop_targets, stop_weights


def audio_and_stop_loss(
    audio_preds: torch.Tensor,
    stop_preds: torch.Tensor,
    audio_targets: torch.Tensor,
    stop_targets: torch.Tensor,
    stop_weights: torch.LongTensor,
    target_mask: torch.BoolTensor,
) -> torch.Tensor:
    """
    计算组合谱图重建损失和停止信号检测损失的加权组合
    该函数主要用于语音合成模型（如Tacotron）的损失计算：
    1. 组合谱图损失：计算预测频谱与目标频谱之间的L1距离（排除填充部分）
    2. 停止信号损失：计算停止信号预测的加权二值交叉熵损失
    3. 最终损失为两项损失的加权和

    Args:
        audio_preds: 模型预测的组合谱图张量，形状为 (B, T, D)
        stop_preds: 停止信号的预测logits张量，形状为 (B, T)
        audio_targets: 真实的组合谱图目标张量，形状为 (B, T + 1, D)
        stop_targets: 真实的停止信号目标张量，形状为 (B, T)
        stop_weights: 停止信号的样本权重张量，形状为 (B, T)
        target_mask: 标识填充位置的布尔掩码张量，形状为 (B, T + 1)

    Returns:
        包含音频损失和停止损失的加权组合的标量张量
    """
    # 计算组合谱图的L1损失（仅计算有效区域）
    # 注意：预测比目标序列少一个时间步（自回归特性）
    audio_loss = F.l1_loss(audio_preds, audio_targets[:, 1:], reduction="none")

    # 创建非填充区域的3D掩码（排除第一个时间步）
    non_padding_mask = ~target_mask[:, 1:].unsqueeze(-1)  # 形状 (B, T, 1)
    non_padding_mask = non_padding_mask.expand_as(audio_loss)  # 扩展至 (B, T, D)

    # 计算有效区域的平均音频损失
    valid_audio_loss = (audio_loss * non_padding_mask).sum()
    valid_audio_loss /= non_padding_mask.sum()

    # 计算停止信号的加权二值交叉熵损失
    stop_loss = F.binary_cross_entropy_with_logits(stop_preds, stop_targets, weight=stop_weights)

    # 返回组合损失（谱图损失 + 停止损失）
    return valid_audio_loss + stop_loss


def train(
    model: TkTTS,
    dataloader: DataLoader,
    optimizer: optim.AdamW,
    scaler: GradScaler,
    accumulation_steps: int = 1,
    device: torch.device = torch.device("cpu")
) -> list[float]:
    """
    训练TkTTS模型的函数，支持梯度累积和混合精度训练

    函数执行流程：
    1. 设置模型为训练模式，初始化梯度缩放器和训练监控工具
    2. 遍历数据加载器中的每个批次
    3. 生成文本和组合谱图的padding mask
    4. 在混合精度环境下计算模型输出和损失
    5. 执行梯度反向传播和参数更新（根据累积步数）
    6. 记录并返回训练过程中的损失值

    Args:
        model: 待训练的TkTTS模型实例
        dataloader: 训练数据加载器
        optimizer: 模型优化器
        scaler: 混合精度梯度缩放器
        accumulation_steps: 梯度累积步数
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
    for step, (text_inputs, audio_targets, text_lengths, audio_lengths) in zip(range(num_iterators), dataloader):
        # 数据移至目标设备
        text_inputs, audio_targets, text_lengths, audio_lengths = text_inputs.to(device), audio_targets.to(device), text_lengths.to(device), audio_lengths.to(device)

        # 准备掩码和停止标志目标
        source_mask, target_mask, stop_targets, stop_weights = prepare_masks_and_targets(text_inputs, audio_targets, text_lengths, audio_lengths, device=device)

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            (audio_preds, stop_preds), _ = model(text_inputs, audio_targets[:, :-1], source_mask, target_mask[:, :-1])  # 模型前向传播（使用教师强制）
            loss = audio_and_stop_loss(audio_preds, stop_preds, audio_targets, stop_targets, stop_weights, target_mask)  # 计算损失

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
    model: TkTTS,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> list[float]:
    # 设置模型为评估模式
    model.eval()

    # 初始化损失列表
    losses = []

    # 遍历整个验证集
    for text_inputs, audio_targets, text_lengths, audio_lengths in tqdm(dataloader, total=len(dataloader)):
        # 数据移至目标设备
        text_inputs, audio_targets, text_lengths, audio_lengths = text_inputs.to(device), audio_targets.to(device), text_lengths.to(device), audio_lengths.to(device)

        # 准备掩码和目标信号
        source_mask, target_mask, stop_targets, stop_weights = prepare_masks_and_targets(text_inputs, audio_targets, text_lengths, audio_lengths, device=device)

        with autocast(device.type, dtype=torch.float16):
            (audio_preds, stop_preds), _ = model(text_inputs, audio_targets[:, :-1], source_mask, target_mask[:, :-1])  # 模型前向传播
            loss = audio_and_stop_loss(audio_preds, stop_preds, audio_targets, stop_targets, stop_weights, target_mask)  # 计算损失

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
    parser.add_argument("-as", "--accumulation-steps", type=int, default=4, help="梯度累积步数，默认为 %(default)s")
    parser.add_argument("-k", "--show-training-process", action="store_true", help="展示训练过程图表，而不仅仅是保存到`<ckpt_path>/statistics.png`")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 设置当前进程的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取检查点
    tokenizer, model_state_dict, generation_config, model_config, optimiezer_state_dict, metrics = load_checkpoint_train(args.ckpt_path)

    # 创建模型并加载状态
    model = TkTTS(model_config, dropout=args.dropout)
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
    train_dataset = TkTTSDataset(args.train_dataset, tokenizer, generation_config["sample_rate"], model_config.fft_length, generation_config["hop_length"], generation_config["win_length"])
    train_sampler = TkTTSDatasetSampler(train_dataset, args.train_max_batch_tokens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=sequence_collate_fn, num_workers=cpu_count())

    # 如果存在验证集，加载验证数据集
    if args.val_dataset:
        val_dataset = TkTTSDataset(args.val_dataset, tokenizer, generation_config["sample_rate"], model_config.fft_length, generation_config["hop_length"], generation_config["win_length"])
        val_sampler = TkTTSDatasetSampler(val_dataset, args.val_max_batch_tokens)
        val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=sequence_collate_fn, num_workers=cpu_count())

    # 开始训练
    for epoch in range(args.num_epochs):
        # 计算累积 Epoch 数
        current_epoch = len(metrics["train_loss"]) + epoch

        # 训练一轮模型
        train_sampler.set_epoch(current_epoch)
        train_loss = train(model, train_loader, optimizer, scaler, accumulation_steps=args.accumulation_steps, device=device)

        # 如果指定了验证集，就进行验证，否则跳过验证并设置验证损失为 NaN
        if args.val_dataset:
            val_sampler.set_epoch(current_epoch)
            val_loss = validate(model, val_loader, device)
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
