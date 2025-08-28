# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
from typing import Optional
import librosa
import orjson
import numpy as np
import pyworld as pw
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.checkpoint import TagLabelEncoder, load_checkpoint
from utils.dataset import AudioMetadata


def convert(
    metadata: list[tuple[pathlib.Path, AudioMetadata]],
    tokenizer: AutoTokenizer,
    tag_label_encoder: TagLabelEncoder,
    sample_rate: int,
    fft_length: int,
    win_length: int,
    hop_length: int,
    num_mels: int,
) -> dict[str, np.ndarray]:
    """
    音频数据处理与特征提取管道，将原始音频转换为训练所需的特征格式并保存

    工作流程：
    1. 遍历音频元数据列表，对每个音频文件进行处理
    2. 使用分词器将文本转换为序列ID
    3. 使用标签编码器处理正负提示词
    4. 加载音频并进行重采样和静音修剪
    5. 提取梅尔频谱、基频和能量特征
    6. 对特征进行归一化处理
    7. 将处理后的特征保存为压缩的NPZ格式文件

    Args:
        metadata: 音频文件路径和对应元数据的列表
        tokenizer: 用于文本编码的分词器
        tag_label_encoder: 用于标签编码的编码器
        sample_rate: 目标采样率
        fft_length: FFT 窗口长度
        win_length: 窗口长度
        hop_length: 跳跃长度
        num_mels: 梅尔带数量

    Returns:
        包含所有处理文件元数据的字典
    """
    # 初始化音频特征字典
    dataset = {}
    text_length = []
    audio_length = []

    # 遍历每一个音频
    for task_id, (audio_path, audio_metadata) in enumerate(tqdm(metadata)):
        # 使用分词器将文本转换为序列 ID
        text_sequences = tokenizer.encode(audio_metadata["text"])

        # 编码正面和负面提示词
        positive_prompt = tag_label_encoder.encode(audio_metadata["positive_prompt"])
        negative_prompt = tag_label_encoder.encode(audio_metadata["negative_prompt"])

        # 加载音频并重采样到目标采样率
        audio, _ = librosa.load(audio_path, sr=sample_rate, dtype=np.float64)

        # 去除音频首尾的静音部分
        audio, _ = librosa.effects.trim(audio)

        # 计算梅尔频谱特征
        mel_spectrogram = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=fft_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=num_mels
        ).T  # 转置以使时间步为第一维度

        # 提取基频
        f0, time_axis = pw.dio(
            audio, sample_rate,
            frame_period=hop_length / sample_rate * 1000
        )

        # 使用 stonemask 修正
        f0_refined = pw.stonemask(audio, f0, time_axis, sample_rate)
        f0_log = np.log(f0_refined + 1e-8)

        # 频谱包络
        sp = pw.cheaptrick(audio, f0_refined, time_axis, sample_rate)

        # 对每一帧的频谱包络求和
        energy = np.sum(sp, axis=1)
        energy_log = np.log(energy + 1e-8)

        # 对音高、能量归一化
        f0_normalized = (f0_log - f0_log.min()) / (f0_log.max() - f0_log.min() + 1e-8)
        energy_normalized = (energy_log - energy_log.min()) / (energy_log.max() - energy_log.min() + 1e-8)

        # 转换数据类型以节省储存空间
        text_sequences = np.array(text_sequences, dtype=int)
        positive_prompt = np.array(positive_prompt, dtype=int)
        negative_prompt = np.array(negative_prompt, dtype=int)
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        f0_normalized = f0_normalized.astype(np.float32)
        energy_normalized = energy_normalized.astype(np.float32)

        # 保存内容到内存
        text_length.append(len(text_sequences))
        audio_length.append(len(mel_spectrogram))
        dataset[f"{task_id}:text"] = text_sequences
        dataset[f"{task_id}:positive_prompt"] = positive_prompt
        dataset[f"{task_id}:negative_prompt"] = negative_prompt
        dataset[f"{task_id}:mel"] = mel_spectrogram
        dataset[f"{task_id}:pitch"] = f0_normalized
        dataset[f"{task_id}:energy"] = energy_normalized

    # 保存长度到内存
    dataset["text_length"] = np.array(text_length)
    dataset["audio_length"] = np.array(audio_length)

    # 返回音频特征字典
    return dataset


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="准备专用于特定检查点训练的数据集，用于加快训练时数据加载速度。注意：为此检查点准备的数据集不能直接用于其他参数配置的检查点训练。")
    parser.add_argument("dataset_metadata", type=pathlib.Path, help="数据集元数据文件的完整路径。该文件应包含音频文件路径及其对应文本和标签信息的 JSON 格式数据")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="目标检查点文件的完整路径。数据集将根据此检查点的配置参数（如分词器、采样率等）进行适配准备")
    parser.add_argument("output_path", type=pathlib.Path, help="处理后的特征数据集输出路径。文件后缀名推荐使用`.npz`标明这是一个 NumPy 数组文件，如`dataset/train.npz`")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 加载检查点
    tokenizer, _, extra_config, model_config, tag_label_encoder = load_checkpoint(args.ckpt_path)

    # 遍历数据集文件元数据
    metadata = [
        (args.dataset_metadata.parent / audio_path, audio_metadata)
        for audio_path, audio_metadata in orjson.loads(args.dataset_metadata.read_bytes()).items()
    ]

    # 打印开始转换消息
    print(f"正在转换 {len(metadata)} 个音频及元数据 ...")

    # 预处理数据
    processed_metadata = convert(metadata, tokenizer, tag_label_encoder, extra_config["sample_rate"], extra_config["fft_length"], extra_config["win_length"], extra_config["hop_length"], model_config.num_mels)

    # 创建输出目录
    args.output_path.parent.mkdir(parents=True, exist_ok=True)

    # 保存到输出路径
    np.savez_compressed(args.output_path, **processed_metadata)

    # 打印保存成功信息
    print(f"预处理完成，快速训练数据集已保存到`{args.output_path}`")


if __name__ == "__main__":
    main(parse_args())
