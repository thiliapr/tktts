# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pathlib
import argparse
from typing import Optional
import torch
import librosa
import soundfile
import numpy as np
from tqdm import tqdm
from utils.checkpoint import load_checkpoint
from utils.model import TkTTS

# 解除线程数量限制
os.environ["OMP_NUM_THREADS"] = os.environ["OPENBLAS_NUM_THREADS"] = os.environ["MKL_NUM_THREADS"] = os.environ["VECLIB_MAXIMUM_THREADS"] = os.environ["NUMEXPR_NUM_THREADS"] = str(os.cpu_count())
torch.set_num_threads(os.cpu_count())


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="根据标签和文本生成一个语音音频")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点路径")
    parser.add_argument("output_path", type=pathlib.Path, help="生成音频文件保存路径")
    parser.add_argument("text", type=str, help="标签和文本，格式: `[标签1][标签2]...[标签N]文本`")
    parser.add_argument("--stop-threshold", type=float, default=0.5, help="音频生成停止概率阈值（0~1）。当模型预测的停止概率超过此值时终止生成。较低的该值会提前结束生成，较高的该值会使生成持续更久。默认：%(default)s")
    parser.add_argument("--max-frames", type=int, default=10000, help="生成的最大帧数。当生成的帧数达到此值时，无论是否满足其他停止条件，都将停止生成。默认值：%(default)s")
    return parser.parse_args(args)


@torch.inference_mode
def main(args: argparse.Namespace):
    # 加载检查点
    tokenizer, model_state, generation_config, model_config = load_checkpoint(args.ckpt_path)

    # 创建模型并加载状态
    model = TkTTS(model_config)
    model.load_state_dict(model_state)

    # 获取设备并将模型转移到设备上
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 设置模型为评估模式
    model.eval()

    # 给输入文本分词并转移到设备上
    source = tokenizer.encode(args.text)
    source = torch.tensor([source], device=device)

    # 创建起始帧
    start_frame = torch.zeros(1, 1, model_config.fft_length * 3 // 2 + 3)

    # 前向传播获取 KV Cache
    (current_frame, _), kv_cache = model(source, start_frame)
    generated_audio = torch.cat([start_frame, current_frame], dim=1)

    # 循环生成音频
    for _ in tqdm(range(args.max_frames)):
        # 生成新的一帧并拼接
        (current_frame, stop_pred), kv_cache = model(target=current_frame, source=None, kv_cache=kv_cache)
        generated_audio = torch.cat([generated_audio, current_frame], dim=1)

        # 判断是否应该结束生成
        if stop_pred.item() > args.stop_threshold:
            break

    # 删除起始帧并去掉批次维度
    generated_audio = generated_audio[0, 1:]

    # 分解为幅度谱、相位cos谱、相位sin谱
    magnitude, phase_cos, phase_sin = generated_audio.chunk(3, dim=-1)

    # 合成相位谱
    phase = torch.atan2(phase_sin, phase_cos)

    # 移动幅度谱、相位谱到 CPU 并转化为 ndarray
    magnitude = magnitude.cpu().numpy()
    phase = phase.cpu().numpy()

    # 转化为 ndarray 并合成 STFT 矩阵
    stft_matrix = magnitude * np.exp(1j * phase)

    # 倒置 STFT 矩阵以转换为符合 librosa.istft 的格式（[num_frame, fft_length // 2 + 1]）
    stft_matrix = stft_matrix.T

    # 逆 STFT 运算
    audio = librosa.istft(stft_matrix, hop_length=generation_config["hop_length"], win_length=generation_config["win_length"], n_fft=model_config.fft_length)

    # 保存为音频
    soundfile.write(args.output_path, audio, generation_config["sample_rate"])


if __name__ == "__main__":
    main(parse_args())
