# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import random
import pathlib
import argparse
from typing import Optional
from collections.abc import Callable, Iterator
import librosa
import torch
import numpy as np
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.amp import GradScaler, autocast
from torch.utils.data import Dataset, Sampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from matplotlib import pyplot as plt
from utils.checkpoint import load_checkpoint_train, save_checkpoint
from utils.constants import DEFAULT_ACCUMULATION_STEPS, DEFAULT_DECODER_DROPOUT, DEFAULT_ENCODER_DROPOUT, DEFAULT_LEARNING_RATE, DEFAULT_POSTNET_DROPOUT, DEFAULT_VARIANCE_PREDICTOR_DROPOUT, DEFAULT_WEIGHT_DECAY, VOICED_THRESHOLD
from utils.model import FastSpeech2
from utils.toolkit import convert_to_tensor, create_padding_mask, get_sequence_lengths

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())


class TkTTSDataset(Dataset):
    """
    文本转语音数据集加载器，用于处理音频-文本配对数据

    Args:
        dataset_file: 快速训练数据集文件

    Yields:
        - 分词后的文本ID序列
        - 正面标签ID序列
        - 负面标签ID序列
        - 对应的梅尔谱图特征矩阵
        - 音频的音高序列
        - 音频的能量序列
    """

    def __init__(self, dataset_file: os.PathLike):
        # 获取所有音频特征
        self.data_samples = np.load(dataset_file)

    def __getitem__(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return (
            self.data_samples[f"{index}:text"],
            self.data_samples[f"{index}:positive_prompt"],
            self.data_samples[f"{index}:negative_prompt"],
            self.data_samples[f"{index}:duration"],
            self.data_samples[f"{index}:mel"],
            self.data_samples[f"{index}:pitch"],
            self.data_samples[f"{index}:energy"]
        )

    def __len__(self) -> int:
        # 每个音频有 7 个特征: 文本序列、正面提示、负面提示、时长比例、梅尔频谱、音高序列、能量序列
        # 然后这个文件还储存了每个文本序列和音频序列的长度，也就是 2 个特征
        # 所以用音频特征总数减去 2 再除以 7 就是音频样本数
        return (len(self.data_samples) - 2) // 7


class TkTTSDatasetSampler(Sampler[list[int]]):
    """
    用于 TkTTS 数据集的分批采样器，根据序列长度进行动态批处理。

    该采样器会:
    1. 根据序列长度对样本进行排序
    2. 动态创建批次，确保每个批次的 token 总数不超过 max_batch_tokens
    3. 每个 epoch 都会重新打乱数据顺序

    Attributes:
        max_batch_tokens: 单个批次允许的最大 token 数量
        seed: 随机种子
        batches: 当前分配到的批次列表

    Examples:
        >>> dataset = TkTTSDataset("dataset.npz")
        >>> sampler = TkTTSDatasetSampler(dataset, max_batch_tokens=4096)
        >>> for batch in sampler:
        ...     print(batch)  # [19, 89, 64]
    """

    def __init__(self, dataset: TkTTSDataset, max_batch_tokens: int, seed: int = 8964):
        super().__init__()
        self.max_batch_tokens = max_batch_tokens
        self.seed = seed
        self.batches: list[list[int]]

        # 预计算所有样本的索引和长度
        text_length = dataset.data_samples["text_length"].tolist()
        audio_length = dataset.data_samples["audio_length"].tolist()
        self.index_and_lengths = [
            # 简单起见，我们用文本序列的长度加上梅尔频谱的帧数作为这个样本的长度，避免一大堆计算
            (idx, text_length[idx] + audio_length[idx])
            for idx in range(len(dataset))
        ]
        self.index_to_length = dict(self.index_and_lengths)

    def set_epoch(self, epoch: int) -> None:
        """
        设置当前 epoch 并重新生成批次

        每个 epoch 开始时调用，用于:
        1. 根据新 epoch 重新打乱数据顺序
        2. 重新分配批次
        """
        generator = random.Random(self.seed + epoch)

        # 按长度排序，加入随机因子避免固定排序
        sorted_pairs = sorted(self.index_and_lengths, key=lambda pair: (pair[1], generator.random()))

        # 初始化批次列表
        self.batches = []
        current_batch = []

        # 遍历每一个样本
        for idx, seq_len in sorted_pairs:
            # 处理超长序列
            if seq_len > self.max_batch_tokens:
                self.batches.append([idx])
                continue

            # 计算当前批次加入新样本后的 token 总数
            estimated_tokens = (len(current_batch) + 1) * seq_len
            if estimated_tokens > self.max_batch_tokens:
                self.batches.append(current_batch)
                current_batch = []

            current_batch.append(idx)

        # 添加最后一个批次
        if current_batch:
            self.batches.append(current_batch)

        # 扰乱批次顺序，增强模型泛化能力
        generator.shuffle(self.batches)

    def __iter__(self) -> Iterator[list[int]]:
        yield from self.batches

    def __len__(self) -> int:
        return len(self.batches)


def sequence_collate_fn(batch: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> tuple[
    torch.LongTensor,
    torch.LongTensor,
    torch.LongTensor,
    torch.BoolTensor,
    torch.BoolTensor,
    torch.BoolTensor,
    torch.LongTensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    处理变长序列数据的批次整理函数
    将输入的多个变长序列样本整理为批次张量，包括文本序列、提示词、时长比例、梅尔频谱图、音高和能量等特征，并生成相应的填充掩码和序列长度信息

    工作流程：
    1. 解压批次数据并将每个特征转换为 PyTorch 张量
    2. 为文本和提示词序列创建填充掩码
    3. 计算音频序列的实际长度
    4. 对所有序列进行填充对齐处理
    5. 返回整理后的批次数据

    Args:
        batch: 包含多个样本的列表，每个样本是包含文本序列、正向提示词、负向提示词、时长比例、梅尔频谱图、音高和能量特征的元组

    Returns:
        包含整理后批次数据的元组，包括填充后的文本序列、正向提示词、负向提示词、
        文本填充掩码、正向提示词掩码、负向提示词掩码、音频长度、时长比例、填充后的音高序列、
        填充后的能量序列和填充后的音频序列

    Examples:
        >>> from torch.utils.data import DataLoader
        >>> dataset = YourDataset()
        >>> dataloader = DataLoader(dataset, batch_size=32, collate_fn=sequence_collate_fn)
        >>> for batch in dataloader:
        >>>     text, pos_prompt, neg_prompt, text_mask, pos_mask, neg_mask, audio_len, duration, pitch, energy, audio = batch
    """
    # 解压批次数据并将每个特征列表转换为张量列表
    text_sequences, positive_prompt, negative_prompt, duration, mel_spectrogram, pitch, energy = [convert_to_tensor(item) for item in zip(*batch)]

    # 创建填充掩码用于标识有效数据位置
    text_padding_mask = create_padding_mask(text_sequences)
    positive_prompt_mask = create_padding_mask(positive_prompt)
    negative_prompt_mask = create_padding_mask(negative_prompt)

    # 计算每个音频样本的实际长度
    audio_length = get_sequence_lengths(mel_spectrogram)

    # 对变长序列进行填充对齐，使批次内所有样本长度一致
    padded_text, padded_positive_prompt, padded_negative_prompt, padded_audio, padded_duration, padded_pitch, padded_energy = [torch.nn.utils.rnn.pad_sequence(item, batch_first=True) for item in [text_sequences, positive_prompt, negative_prompt, mel_spectrogram, duration, pitch, energy]]

    # 返回批次数据
    return (
        padded_text,
        padded_positive_prompt,
        padded_negative_prompt,
        text_padding_mask,
        positive_prompt_mask,
        negative_prompt_mask,
        audio_length,
        padded_duration,
        padded_pitch,
        padded_energy,
        padded_audio
    )


def fastspeech2_loss(
    postnet_pred: torch.Tensor,
    audio_pred: torch.Tensor,
    duration_pred: torch.Tensor,
    pitch_pred: torch.Tensor,
    energy_pred: torch.Tensor,
    audio_target: torch.Tensor,
    duration_target: torch.Tensor,
    pitch_target: torch.Tensor,
    energy_target: torch.Tensor,
    text_padding_mask: torch.BoolTensor,
    audio_padding_mask: torch.BoolTensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    计算 FastSpeech2 模型的复合损失函数，包含音频重建、时长、音高和能量损失。

    通过生成掩码来识别有效音频帧，分别计算各分量的L1损失，并对有效区域求平均。
    时长损失使用总时长比较，其他损失使用逐帧掩码计算，最后返回加权和。

    Args:
        postnet_pred: 经过后处理网络的梅尔频谱预测，形状为 [batch_size, audio_len, num_mels]
        audio_pred: 预测的原始梅尔频谱图，形状为 [batch_size, audio_len, num_mels]
        duration_pred: 预测的时长特征，形状为 [batch_size, text_len]
        pitch_pred: 预测的音高特征，形状为 [batch_size, audio_len]
        energy_pred: 预测的能量特征，形状为 [batch_size, audio_len]
        audio_target: 真实的梅尔频谱图，形状为 [batch_size, audio_len, num_mels]
        duration_target: 真实的时长特征，形状为 [batch_size, text_len]
        pitch_target: 真实的音高特征，形状为 [batch_size, audio_len]
        energy_target: 真实的能量特征，形状为 [batch_size, audio_len]
        audio_padding_mask: 音频预测填充掩码，形状为 [batch_size, audio_len]

    Returns:
        每个序列的各分量（后处理网络、原始梅尔频谱、时长、音高、能量）损失值，标量张量
    """
    def masked_loss(pred: torch.Tensor, target: torch.Tensor, padding_mask: torch.BoolTensor, criterion: Callable[..., torch.Tensor]):
        "计算掩码区域的损失，仅对有效帧求平均"
        # 截断长度
        min_seq_len = min(pred.size(1), target.size(1))
        pred = pred[:, :min_seq_len]
        target = target[:, :min_seq_len]
        padding_mask = padding_mask[:, :min_seq_len]

        # 计算逐元素L1损失 [batch_size, seq_len, ...]
        elementwise_loss = criterion(pred, target, reduction="none")

        # 重塑损失张量以便统一 duration/pitch/energy 和 mel 掩码应用逻辑
        # [batch_size, seq_len, feature_dim]，其中 feature_dim 可能为 1 或 dim_model
        loss_reshaped = elementwise_loss.view(*elementwise_loss.shape[:2], -1)

        # 扩展掩码以匹配损失张量的形状
        expanded_mask = padding_mask.unsqueeze(2).expand_as(loss_reshaped)

        # 将填充区域的损失置零
        masked_loss = loss_reshaped.masked_fill(expanded_mask, 0)

        # 计算损失平均值
        return masked_loss.sum(dim=[1, 2]) / (~expanded_mask).sum(dim=[1, 2])

    # 计算各分量损失
    postnet_loss = masked_loss(postnet_pred, audio_target, audio_padding_mask, F.l1_loss)
    audio_loss = masked_loss(audio_pred, audio_target, audio_padding_mask, F.l1_loss)
    duration_loss = masked_loss(duration_pred, duration_target, text_padding_mask, F.mse_loss)
    pitch_loss = masked_loss(pitch_pred, pitch_target, audio_padding_mask, F.mse_loss)
    energy_loss = masked_loss(energy_pred, energy_target, audio_padding_mask, F.mse_loss)

    # 返回各分量损失
    return postnet_loss, audio_loss, duration_loss, pitch_loss, energy_loss


def train(
    model: FastSpeech2,
    dataloader: DataLoader,
    optimizer: optim.AdamW,
    scaler: GradScaler,
    completed_iters: int,
    writer: SummaryWriter,
    accumulation_steps: int = 1,
    device: torch.device = torch.device("cpu")
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    训练 TkTTS-FastSpeech2 模型的函数，支持梯度累积和混合精度训练

    Args:
        model: 待训练的 TkTTS 模型实例
        dataloader: 训练数据加载器
        optimizer: 模型优化器
        scaler: 混合精度梯度缩放器
        completed_iters: 已经完成多少次迭代，记录损失用
        writer: 记录训练损失用的
        accumulation_steps: 梯度累积步数
        device: 训练设备（默认使用 CPU）

    Returns:
        最后一步的梅尔频谱、时长比例、音高、能量预测和目标
    """
    # 设置模型为训练模式
    model.train()

    # 计算前向传播多少次
    num_steps = len(dataloader) // accumulation_steps * accumulation_steps

    # 创建进度条，显示训练进度
    progress_bar = tqdm(total=num_steps, desc="Train")

    # 当前累积步骤的总损失
    acc_postnet_loss = acc_audio_loss = acc_duration_loss = acc_pitch_loss = acc_energy_loss = 0

    # 提前清空梯度
    optimizer.zero_grad()

    # 遍历整个训练集
    for step, batch in zip(range(num_steps), dataloader):
        # 数据移至目标设备
        text_sequences, positive_prompt, negative_prompt, text_padding_mask, positive_prompt_mask, negative_prompt_mask, audio_length, duration_target, pitch_target, energy_target, audio_target = [(item.to(device=device) if isinstance(item, torch.Tensor) else item) for item in batch]

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            postnet_pred, audio_pred, duration_pred, pitch_pred, energy_pred, audio_padding_mask = model(text_sequences, audio_length, positive_prompt, negative_prompt, text_padding_mask, positive_prompt_mask, negative_prompt_mask, duration_target, pitch_target, energy_target)  # 模型前向传播（使用教师强制）
            all_loss = fastspeech2_loss(postnet_pred, audio_pred, duration_pred, pitch_pred, energy_pred, audio_target, duration_target, pitch_target, energy_target, text_padding_mask, audio_padding_mask)  # 计算损失
            postnet_loss, audio_loss, duration_loss, pitch_loss, energy_loss = (loss.mean() for loss in all_loss)  # 计算整个批次的损失
            value = postnet_loss + audio_loss + duration_loss + pitch_loss + energy_loss

        # 梯度缩放与反向传播
        scaler.scale(value).backward()

        # 更新累积损失
        acc_postnet_loss += postnet_loss.item() / accumulation_steps
        acc_audio_loss += audio_loss.item() / accumulation_steps
        acc_duration_loss += duration_loss.item() / accumulation_steps
        acc_pitch_loss += pitch_loss.item() / accumulation_steps
        acc_energy_loss += energy_loss.item() / accumulation_steps

        # 达到累积步数时更新参数
        if (step + 1) % accumulation_steps == 0:
            scaler.step(optimizer)  # 更新模型参数
            scaler.update()  # 调整缩放因子
            optimizer.zero_grad()  # 清空梯度
            global_step = completed_iters + ((step + 1) // accumulation_steps) - 1  # 计算全局步数

            # 记录损失
            for loss_name, loss_value in [
                ("Post-Net", acc_postnet_loss),
                ("Mel", acc_audio_loss),
                ("Duration", acc_duration_loss),
                ("Pitch", acc_pitch_loss),
                ("Energy", acc_energy_loss),
            ]:
                writer.add_scalars(f"Loss/{loss_name}", {"Train": loss_value}, global_step)

            # 记录缩放
            for name, value in [
                ("Scale/Post-Net", model.postnet_scale.item()),
                *[
                    (f"Scale/{layer_type}.{layer_idx}.{module_name}", module_scale.item())
                    for layer_type, layers in [("Encoder", model.encoder), ("Decoder", model.decoder)]
                    for layer_idx, layer in enumerate(layers)
                    for module_name, module_scale in [("Attention", layer.attention_scale), ("FeedForward", layer.feedforward_scale)]
                ]
            ]:
                writer.add_scalar(name, value, global_step)

            # 重置累积损失
            acc_postnet_loss = acc_audio_loss = acc_duration_loss = acc_pitch_loss = acc_energy_loss = 0

        # 更新进度条
        progress_bar.update()

        # 最后一步时，记录训练集上预测和目标值
        if step + 1 == num_steps:
            sample_length = audio_length[0].item()
            results = [
                x[0, :sample_length].detach().cpu().numpy()
                for x in [postnet_pred, audio_target, duration_pred, duration_target, pitch_pred, pitch_target, energy_pred, energy_target]
            ]
            return results[::2], results[1::2]


@torch.inference_mode
def validate(
    model: FastSpeech2,
    dataloader: DataLoader,
    device: torch.device = torch.device("cpu")
) -> list[tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], tuple[float, float, float, float, float]]]:
    """
    在验证集上评估 FastSpeech2 模型的性能
    计算模型在验证集上的各项损失值，包括音频重建损失、音高损失和能量损失
    使用推理模式禁用梯度计算，节省内存并加速验证过程
    自动处理音频序列长度不匹配的问题，通过填充或截断确保损失计算的正确性
    支持自动混合精度计算，在保持精度的同时提升计算效率

    Args:
        model: 要验证的 FastSpeech2 模型实例
        dataloader: 验证集的数据加载器，提供批次数据
        device: 计算设备，用于指定模型和数据所在的硬件设备

    Returns:
        包含每个样本详细损失信息的列表，每个元素为元组：
        (预测, 目标, (后处理音频损失, 原始音频损失, 时长比例损失, 音高损失, 能量损失))

    Examples:
        >>> validation_results = validate(model, val_loader, 1.0, torch.device("cuda"))
        >>> audio_loss_avg = sum(result[3][0] for result in validation_results) / len(validation_results)
    """
    # 设置模型为评估模式
    model.eval()

    # 初始化损失列表
    loss_results = []

    # 遍历验证集所有批次数据，显示进度条
    for batch in tqdm(dataloader, total=len(dataloader), desc="Validate"):
        # 数据移至目标设备
        text_sequences, positive_prompt, negative_prompt, text_padding_mask, positive_prompt_mask, negative_prompt_mask, audio_length, duration_target, pitch_target, energy_target, audio_target = [(item.to(device=device) if isinstance(item, torch.Tensor) else item) for item in batch]

        # 自动混合精度环境
        with autocast(device.type, dtype=torch.float16):
            # 模型前向传播（不使用教师强制）
            postnet_pred, audio_pred, duration_pred, pitch_pred, energy_pred, audio_padding_mask = model(text_sequences, audio_length, positive_prompt, negative_prompt, text_padding_mask, positive_prompt_mask, negative_prompt_mask)

            # 计算损失
            all_loss = fastspeech2_loss(postnet_pred, audio_pred, duration_pred, pitch_pred, energy_pred, audio_target, duration_target, pitch_target, energy_target, text_padding_mask, audio_padding_mask)

        # 记录当前批次的损失信息和预测-目标值
        for seq_idx in range(text_sequences.size(0)):
            sample_length = audio_length[seq_idx].item()
            results = [
                x[seq_idx, :sample_length].cpu().numpy()
                for x in [postnet_pred, audio_target, duration_pred, duration_target, pitch_pred, pitch_target, energy_pred, energy_target]
            ]
            loss_results.append((
                results[::2],
                results[1::2],
                tuple(loss[seq_idx].item() for loss in all_loss)
            ))

    return loss_results


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
    parser.add_argument("-t", "--train-dataset", type=pathlib.Path, required=True, help="训练集文件路径")
    parser.add_argument("-v", "--val-dataset", type=pathlib.Path, required=True, help="验证集文件路径")
    parser.add_argument("-tt", "--train-max-batch-tokens", default=16384, type=int, help="训练时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-tv", "--val-max-batch-tokens", default=32768, type=int, help="验证时，每个批次的序列长度的和上限，默认为 %(default)s")
    parser.add_argument("-lr", "--learning-rate", default=DEFAULT_LEARNING_RATE, type=float, help="学习率，默认为 %(default)s")
    parser.add_argument("-wd", "--weight-decay", default=DEFAULT_WEIGHT_DECAY, type=float, help="权重衰减系数，默认为 %(default)s")
    parser.add_argument("-de", "--encoder-dropout", default=DEFAULT_ENCODER_DROPOUT, type=float, help="编码器 Dropout 概率，用于防止过拟合，默认为 %(default)s")
    parser.add_argument("-dd", "--decoder-dropout", default=DEFAULT_DECODER_DROPOUT, type=float, help="解码器 Dropout 概率，用于防止过拟合，默认为 %(default)s")
    parser.add_argument("-dv", "--variance-predictor-dropout", default=DEFAULT_VARIANCE_PREDICTOR_DROPOUT, type=float, help="变异性预测器 Dropout 概率，用于防止过拟合，默认为 %(default)s")
    parser.add_argument("-dp", "--postnet-dropout", default=DEFAULT_POSTNET_DROPOUT, type=float, help="后处理网络 Dropout 概率，用于防止过拟合，默认为 %(default)s")
    parser.add_argument("-as", "--accumulation-steps", default=DEFAULT_ACCUMULATION_STEPS, type=int, help="梯度累积步数，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 设置当前进程的设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 读取检查点
    print("读取检查点 ...")
    model_state_dict, extra_config, model_config, tag_label_encoder, optimiezer_state_dict, completed_epochs = load_checkpoint_train(args.ckpt_path)

    # 创建模型并加载状态
    model = FastSpeech2(model_config, args.encoder_dropout, args.decoder_dropout, args.variance_predictor_dropout, args.postnet_dropout)
    model.load_state_dict(model_state_dict)

    # 转移模型到设备
    model = model.to(device)

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
    print("加载训练集 ...")
    train_dataset = TkTTSDataset(args.train_dataset)
    train_sampler = TkTTSDatasetSampler(train_dataset, args.train_max_batch_tokens)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 加载验证数据集
    print("加载验证集 ...")
    val_dataset = TkTTSDataset(args.val_dataset)
    val_sampler = TkTTSDatasetSampler(val_dataset, args.val_max_batch_tokens)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=sequence_collate_fn, num_workers=0)

    # 创建一个 SummaryWriter 实例，用于记录训练过程中的指标和可视化数据
    writer = SummaryWriter(args.ckpt_path / f"logdir/default")

    # 开始训练
    for epoch in range(args.num_epochs):
        # 计算累积 Epoch 数
        current_epoch = completed_epochs + epoch

        # 训练一轮模型
        train_sampler.set_epoch(current_epoch)
        train_pred, train_target = train(model, train_loader, optimizer, scaler, len(train_loader) // args.accumulation_steps * current_epoch, writer, args.accumulation_steps, device=device)

        # 验证模型效果
        val_sampler.set_epoch(current_epoch)
        val_results = validate(model, val_loader, device)

        # 选择训练集最后一步预测，和验证集随机选择结果，绘制预测频谱图和其目标频谱图
        val_pred, val_target, _ = random.choice(val_results)
        for title, pred, target in [
            ("Train Sample", train_pred, train_target),
            ("Validate Sample", val_pred, val_target),
        ]:
            # 创建图像和坐标轴
            figure, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

            # 绘制对比图
            for mel_axes, figure_title, mel, duration, pitch, energy in [
                (ax1, "Predicted Mel Spectrogram", *pred),
                (ax2, "True Mel Spectrogram", *target)
            ]:
                # 倒置为 [num_mels, num_frames] 维度，展示梅尔频谱
                librosa.display.specshow(mel.T, y_axis="mel", x_axis="time", sr=extra_config["sample_rate"], fmax=8000, ax=mel_axes)

                # 将清音部分绘制为空白
                pitch[pitch < VOICED_THRESHOLD] = np.nan

                # 限制音高、能量范围
                pitch[~np.isnan(pitch)] = np.clip(pitch[~np.isnan(pitch)], 0, 1)
                energy = np.clip(energy, 0, 1)

                # 绘制音高和能量
                times = librosa.times_like(pitch, sr=extra_config["sample_rate"])
                normalized_axes = mel_axes.twinx()
                normalized_axes.set_ylim(0, 1)
                normalized_axes.plot(times, pitch, label="Pitch", color="blue")
                normalized_axes.plot(times, energy, label="Energy", color="red")

                # 绘制时长比例
                phoneme_positions = np.cumsum(duration)
                for position in phoneme_positions:
                    mel_axes.axvline(min(position, 1) * times[-1])

                # 设置标题并创造图例
                mel_axes.set_title(figure_title)
                normalized_axes.legend()

            # 添加图像到 writer
            writer.add_figure(f"Epoch {current_epoch + 1}/{title}", figure)

        # 绘制验证损失分布直方图，记录验证损失
        for loss_idx, loss_name in enumerate(["Post-Net", "Mel", "Duration", "Pitch", "Energy"]):
            loss_values = [all_loss[loss_idx] for _, _, all_loss in val_results]
            writer.add_histogram(f"Epoch {current_epoch + 1}/Validate/{loss_name} Loss Distribution", np.array(loss_values))
            writer.add_scalars(f"Loss/{loss_name}", {"Valid": np.array(loss_values).mean()}, len(train_loader) // args.accumulation_steps * (current_epoch + 1))

    # 记录模型的文本和标签嵌入，覆盖上一个记录
    writer.add_embedding(model.tag_embedding.weight, [tag_label_encoder.id_to_tag[token_id] for token_id in range(len(tag_label_encoder))], tag=f"Tag Embedding")

    # 关闭 SummaryWriter 实例，确保所有记录的数据被写入磁盘并释放资源
    writer.close()

    # 保存当前模型的检查点
    save_checkpoint(
        args.ckpt_path,
        model.cpu().state_dict(),
        optimizer.state_dict(),
        completed_epochs + args.num_epochs
    )
    print(f"训练完成，模型已保存到 {args.ckpt_path}，训练过程记录保存到 {args.ckpt_path / 'logdir'}，你可以通过 `tensorboard --logdir {args.ckpt_path / 'logdir'}` 查看。")


if __name__ == "__main__":
    main(parse_args())
