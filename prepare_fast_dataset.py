# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import argparse
import pathlib
from typing import Optional
import librosa
import orjson
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from utils.checkpoint import TagLabelEncoder, load_checkpoint
from utils.dataset import AudioMetadata
from utils.tookit import parallel_map


def convert_and_save(
    rank: int,
    metadata: list[tuple[pathlib.Path, AudioMetadata]],
    output_dir: pathlib.Path,
    tokenizer: AutoTokenizer,
    tag_label_encoder: TagLabelEncoder,
    sample_rate: int,
    fft_length: int,
    frame_length: int,
    win_length: int,
    hop_length: int,
    num_mels: int,
) -> dict[str, dict[str, list[int]]]:
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
    8. 构建并返回训练数据集的元信息字典

    Args:
        rank: 当前进程的排名（用于分布式训练）
        metadata: 音频文件路径和对应元数据的列表
        output_dir: 处理后的特征文件输出目录
        tokenizer: 用于文本编码的分词器
        tag_label_encoder: 用于标签编码的编码器
        sample_rate: 目标采样率
        fft_length: FFT 窗口长度
        frame_length: 帧长度
        win_length: 窗口长度
        hop_length: 跳跃长度
        num_mels: 梅尔带数量

    Returns:
        包含所有处理文件元数据的字典
    """
    # 初始化数据集元数据、内容分块
    dataset_metadata = []
    dataset_chunk = {}

    # 仅在主进程中显示进度条
    for task_id, (audio_path, audio_metadata) in enumerate(tqdm(metadata, disable=rank != 0)):
        # 使用分词器将文本转换为序列 ID
        text_sequences = tokenizer.encode(audio_metadata["text"])

        # 编码正面和负面提示词
        positive_prompt = tag_label_encoder.encode(audio_metadata["positive_prompt"])
        negative_prompt = tag_label_encoder.encode(audio_metadata["negative_prompt"])

        # 加载音频并重采样到目标采样率
        audio, _ = librosa.load(audio_path, sr=sample_rate)

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

        # 设置基频提取的参数范围
        fmin = librosa.note_to_hz("C2")  # 最低频率（男性~65Hz，女性~100Hz）
        fmax = librosa.note_to_hz("C7")  # 最高频率

        # 使用 YIN 算法提取基频
        f0 = librosa.yin(
            audio,
            fmin=fmin,
            fmax=fmax,
            sr=sample_rate,
            frame_length=frame_length,
            hop_length=hop_length,
        )

        # 添加微小值避免对数运算错误
        f0 += np.finfo(f0.dtype).eps

        # 对基频进行对数变换和归一化
        log_f0 = np.log(f0)
        log_min = np.log(fmin)
        log_max = np.log(fmax)
        normalized_f0 = (log_f0 - log_min) / (log_max - log_min)

        # 计算短时能量（RMS）
        energy = librosa.feature.rms(
            y=audio,
            frame_length=frame_length,
            hop_length=hop_length,
            center=True,
        ).squeeze(0)  # 移除声道维度

        # 将能量转换为分贝单位
        energy_db = librosa.amplitude_to_db(energy)

        # 对能量进行归一化处理
        normalized_energy = (energy_db - energy_db.min()) / (energy_db.max() - energy_db.min())

        # 转换数据类型以节省储存空间
        mel_spectrogram = mel_spectrogram.astype(np.float32)
        normalized_f0 = normalized_f0.astype(np.float32)
        normalized_energy = normalized_energy.astype(np.float32)

        # 保存元数据信息
        dataset_metadata.append({
            "text": text_sequences,
            "positive_prompt": positive_prompt,
            "negative_prompt": negative_prompt,
            "rank": rank,
            "audio_id": task_id
        })

        # 保存内容
        dataset_chunk[":".join(task_id, "mel")] = mel_spectrogram
        dataset_chunk[":".join(task_id, "pitch")] = normalized_f0
        dataset_chunk[":".join(task_id, "energy")] = normalized_energy

    # 将内容分块写入文件
    np.savez_compressed(output_dir / f"dataset-{rank}.npz", **dataset_chunk)
    return dataset_metadata


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
    parser.add_argument("output_dir", type=pathlib.Path, help="处理后的特征数据集输出目录路径。所有提取的音频特征和转换后的文本序列将保存在此目录中")
    parser.add_argument("-f", "--metadata-filename", type=str, default="metadata.json", help="处理后的特征数据集的元数据文件名，默认为 %(default)s")
    parser.add_argument("-n", "--num-workers", type=int, default=os.cpu_count(), help="并行处理任务的工作进程数量。默认值为当前系统的 CPU 核心数 %(default)s")
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

    # 并行转换数据
    results = parallel_map(convert_and_save, [
        (rank, metadata[rank::args.num_workers], args.output_dir, tokenizer, tag_label_encoder, extra_config["sample_rate"], extra_config["fft_length"], extra_config["frame_length"], extra_config["win_length"], extra_config["hop_length"], model_config.num_mels)
        for rank in range(args.num_workers)
    ])

    # 合并元数据
    processed_metadata = {k: v for worker_metadata in results for k, v in worker_metadata.items()}

    # 保存到输出路径
    (args.output_dir / args.metadata_filename).write_bytes(orjson.dumps(processed_metadata))

    # 打印保存成功信息
    print(f"预处理完成，快速训练数据集已保存到`{args.output_dir}`，其中`{args.metadata_filename}`是该数据集的元数据文件。")


if __name__ == "__main__":
    main(parse_args())
