# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
import pathlib
import argparse
from turtle import forward
from typing import Optional
import torch
import librosa
import soundfile
from utils.checkpoint import load_checkpoint
from utils.model import FastSpeech2

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
    parser.add_argument("text", type=str, help="文本")
    parser.add_argument("-p", "--positive-prompt", type=str, action="append", help="正面提示词，可以是多个词")
    parser.add_argument("-n", "--negative-prompt", type=str, action="append", help="负面提示词，可以是多个词")
    return parser.parse_args(args)


@torch.inference_mode
def main(args: argparse.Namespace):
    # 加载检查点
    tokenizer, model_state, extra_config, model_config, tag_label_encoder = load_checkpoint(args.ckpt_path)

    # 创建模型并加载状态
    model = FastSpeech2(model_config)
    model.load_state_dict(model_state)

    # 获取设备并将模型转移到设备上
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # 设置模型为评估模式
    model.eval()

    # 给输入文本分词
    text = tokenizer.encode(args.text)

    # 打印分词效果
    print(f"分词后的文本: {tokenizer.decode(text)}")

    # 将分词后的文本转换为张量并转移到设备上
    text = torch.tensor([text], device=device)

    # 创建正面和负面提示词的编码
    positive_prompt = tag_label_encoder.encode(args.positive_prompt) if args.positive_prompt else None
    negative_prompt = tag_label_encoder.encode(args.negative_prompt) if args.negative_prompt else None

    # 打印有效的正面和负面提示词
    if positive_prompt is not None:
        print(f"正面提示词: {[tag_label_encoder.id_to_tag[tag] for tag in positive_prompt]}")
    if negative_prompt is not None:
        print(f"负面提示词: {[tag_label_encoder.id_to_tag[tag] for tag in negative_prompt]}")

    # 给提示词加上批次维度
    positive_prompt = [positive_prompt] if positive_prompt else None
    negative_prompt = [negative_prompt] if negative_prompt else None

    # 生成音频
    mel_prediction, _, _, _ = model(text, positive_prompt, negative_prompt)  # [1, seq_len, n_mels]

    # 打印帧数
    print(f"STFT 帧数: {len(mel_prediction)}")

    # 将生成的梅尔频谱转换为 STFT 矩阵
    stft_matrix = librosa.feature.inverse.mel_to_stft(
        mel_prediction.squeeze(0).cpu().numpy().T,  # 转置为 [num_mels, seq_len]
        sr=extra_config["sample_rate"],
        n_fft=extra_config["fft_length"],
    )

    # 逆 STFT 运算
    audio = librosa.istft(stft_matrix, hop_length=extra_config["hop_length"], win_length=extra_config["win_length"], n_fft=extra_config["fft_length"])

    # 打印时长
    print(f"生成音频时长: {len(audio) / extra_config['sample_rate']:.3f} 秒")

    # 保存为音频
    soundfile.write(args.output_path, audio, extra_config["sample_rate"])
    print(f"音频文件已保存到 {args.output_path}")


if __name__ == "__main__":
    main(parse_args())
