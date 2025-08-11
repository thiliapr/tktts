# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
from typing import Optional
import torch
import numpy as np
from torch import optim
from transformers import AutoTokenizer
from utils.checkpoint import save_checkpoint
from utils.constants import DEFAULT_FFT_LENGTH, DEFAULT_HOP_LENGTH, DEFAULT_NUM_HEADS, DEFAULT_DIM_HEAD, DEFAULT_DIM_FEEDFORWARD, DEFAULT_NUM_LAYERS, DEFAULT_SAMPLE_RATE, DEFAULT_WIN_LENGTH
from utils.model import TkTTS, TkTTSConfig


def set_seed(seed: int):
    """
    设置所有随机源的种子以确保实验可复现性。

    工作流程:
    1. 设置Python内置random模块的种子
    2. 设置NumPy的随机种子
    3. 设置PyTorch的CPU和GPU随机种子
    4. 配置CuDNN使用确定性算法并关闭benchmark模式

    Args:
        seed: 要设置的随机种子值

    Examples:
        >>> set_seed(8964)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # 多GPU情况
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="初始化一个检查点")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点保存目录路径")
    parser.add_argument("-sr", "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="目标采样率(Hz)，默认为 %(default)s")
    parser.add_argument("-fl", "--fft-length", type=int, default=DEFAULT_FFT_LENGTH, help="FFT窗口长度，默认为 %(default)s")
    parser.add_argument("-hl", "--hop-length", type=int, default=DEFAULT_HOP_LENGTH, help="帧移长度，默认为 %(default)s")
    parser.add_argument("-wl", "--win-length", type=int, default=DEFAULT_WIN_LENGTH, help="窗函数长度，默认为 %(default)s")
    parser.add_argument("-nh", "--num-heads", type=int, default=DEFAULT_NUM_HEADS, help="注意力头的数量，默认为 %(default)s")
    parser.add_argument("-dh", "--dim-head", type=int, default=DEFAULT_DIM_HEAD, help="每个注意力头的维度，默认为 %(default)s")
    parser.add_argument("-df", "--dim-feedforward", type=int, default=DEFAULT_DIM_FEEDFORWARD, help="前馈网络的隐藏层维度，默认为 %(default)s")
    parser.add_argument("-nl", "--num-layers", type=int, default=DEFAULT_NUM_LAYERS, help="Transformer En/Decoder 层的数量，默认为 %(default)s")
    parser.add_argument("-u", "--seed", default=8964, type=int, help="初始化检查点的种子，保证训练过程可复现，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 检查 FFT 窗口长度是否为偶数
    if args.fft_length % 2 == 1:
        raise RuntimeError("FFT 窗口长度必须为偶数。")

    # 设置随机种子，确保可复现性
    set_seed(args.seed)

    # 加载分词器
    tokenizer_path = (args.ckpt_path / "tokenizer")
    if not tokenizer_path.exists():
        raise RuntimeError("你应该先训练分词器再初始化检查点，因为创建模型模型需要提供分词器的大小。")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 初始化模型
    model = TkTTS(TkTTSConfig(
        vocab_size=len(tokenizer),
        fft_length=args.fft_length,
        num_heads=args.num_heads,
        dim_head=args.dim_head,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers
    ))

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters())

    # 初始化训练历史记录
    metrics = {"val_loss": [], "train_loss": []}

    # 生成模型生成配置
    generation_config = {
        "num_heads": args.num_heads,
        "sample_rate": args.sample_rate,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
    }

    # 保存为检查点
    save_checkpoint(args.ckpt_path, model.state_dict(), optimizer.state_dict(), metrics, generation_config)


if __name__ == "__main__":
    main(parse_args())
