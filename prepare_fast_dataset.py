# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
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
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """
    音频数据处理与特征提取管道，将原始音频转换为训练所需的特征格式并保存

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
        包含所有处理文件元数据的列表、每个文件的文本序列长度、音频序列长度
    """
    # 初始化音频特征列表
    dataset = []

    # 遍历每一个音频
    for audio_path, audio_metadata in tqdm(metadata):
        # 使用分词器将文本转换为序列 ID
        text = tokenizer.encode(audio_metadata["text"])

        # 编码正面和负面提示词
        positive_prompt = tag_label_encoder.encode(audio_metadata["positive_prompt"])
        negative_prompt = tag_label_encoder.encode(audio_metadata["negative_prompt"])

        # 加载音频并重采样到目标采样率
        audio, _ = librosa.load(audio_path, sr=sample_rate, dtype=np.float64)

        # 去除音频首尾的静音部分
        audio, _ = librosa.effects.trim(audio)

        # 提取基频
        f0, time_axis = pw.dio(
            audio, sample_rate,
            frame_period=hop_length / sample_rate * 1000
        )

        # 使用 stonemask 修正
        f0 = pw.stonemask(audio, f0, time_axis, sample_rate)

        # 频谱包络
        sp = pw.cheaptrick(audio, f0, time_axis, sample_rate)

        # 对每一帧的频谱包络求和
        energy = np.sum(sp, axis=1)
        energy = np.log(energy + 1e-8)

        # 计算梅尔频谱特征
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sample_rate,
            n_fft=fft_length,
            hop_length=hop_length,
            win_length=win_length,
            n_mels=num_mels
        ).T  # 转置以使时间步为第一维度

        # 转换为分贝尺度
        mel = librosa.power_to_db(mel, ref=np.max)

        # 截取梅尔频谱，使梅尔频谱、音高、能量长度匹配
        mel = mel[:len(f0)]

        # 对梅尔频谱逐样本归一化
        mel = (mel - mel.min()) / (mel.max() - mel.min() + 1e-8)

        # 转换数据类型以节省储存空间
        text, positive_prompt, negative_prompt = (np.array(x, dtype=int) for x in [text, positive_prompt, negative_prompt])
        mel, f0, energy = (x.astype(np.float32) for x in [mel, f0, energy])

        # 保存内容到内存
        dataset.append([text, positive_prompt, negative_prompt, mel, f0, energy])

    # 计算音高、能量的百分位数
    pitch_min, pitch_max = np.percentile(np.concatenate([pitch for _, _, _, _, pitch, _ in dataset]), [1, 99])
    energy_min, energy_max = np.percentile(np.concatenate([energy for _, _, _, _, _, energy in dataset]), [1, 99])

    # 归一化音高、能量
    for idx, (_, _, _, _, pitch, energy) in enumerate(dataset):
        dataset[idx][-1] = np.clip(((energy - energy_min) / (energy_max - energy_min)), 0, 1).astype(np.float32)
        dataset[idx][-2] = np.clip(((pitch - pitch_min) / (pitch_max - pitch_min)), -np.inf, 1).astype(np.float32)
        dataset[idx][-2][pitch <= 0] = -1

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
    parser.add_argument("output_dir", type=pathlib.Path, help="处理后的特征数据集输出目录")
    parser.add_argument("splits", type=str, action="append", help="输出文件名和拆分比例，格式为`filename:proportion`，如`train:9`和`val:1`")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 加载检查点
    tokenizer, _, extra_config, model_config, tag_label_encoder = load_checkpoint(args.ckpt_path)

    # 遍历数据集文件元数据
    metadata = [
        (args.dataset_metadata.parent / audio_path, audio_metadata)
        for audio_path, audio_metadata in orjson.loads(args.dataset_metadata.read_bytes()).items()
    ]

    # 解析拆分配置
    splits = [str_split.split(":", 1) for str_split in args.splits]
    splits = [(filename, int(proportion)) for filename, proportion in splits]
    total_proportion = sum(proportion for _, proportion in splits)

    # 验证数据集大小是否满足拆分需求
    if len(metadata) < total_proportion:
        raise RuntimeError(f"音频数据数量（{len(metadata)}）小于指定的拆分比例总和（{total_proportion}）")

    # 打印开始转换消息
    print(f"正在转换 {len(metadata)} 个音频及元数据 ...")

    # 预处理数据
    processed_metadata = convert(metadata, tokenizer, tag_label_encoder, extra_config["sample_rate"], extra_config["fft_length"], extra_config["win_length"], extra_config["hop_length"], model_config.num_mels)

    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 随机打乱元数据
    random.shuffle(processed_metadata)

    # 按比例拆分数据
    split_data = [processed_metadata[rank::total_proportion] for rank in range(total_proportion)]

    # 写入拆分后的文件
    for filename, proportion in splits:
        # 从拆分数据中提取对应比例的数据
        subset = [item for chunk in split_data[:proportion] for item in chunk]
        split_data = split_data[proportion:]

        # 转换为字典形式
        data = {}
        text_length = []
        audio_length = []
        for task_id, (text, positive_prompt, negative_prompt, mel, pitch, energy) in enumerate(subset):
            data |= {
                f"{task_id}:text": text,
                f"{task_id}:positive_prompt": positive_prompt,
                f"{task_id}:negative_prompt": negative_prompt,
                f"{task_id}:mel": mel,
                f"{task_id}:pitch": pitch,
                f"{task_id}:energy": energy,
            }
            text_length.append(len(text))
            audio_length.append(len(mel))

        # 将子集写入对应文件
        np.savez_compressed(args.output_dir / f"{filename}.npz", **data, audio_length=np.array(audio_length), text_length=np.array(text_length))
        print(f"数据集的 {proportion}/{total_proportion}，即 {len(subset)} 条数据，已保存到 {filename}.npz")


if __name__ == "__main__":
    main(parse_args())
