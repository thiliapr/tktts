# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import shutil
import warnings
import argparse
import pathlib
import random
from typing import Optional
import torch
import numpy as np
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer
from utils.checkpoint import TagLabelEncoder, save_checkpoint
from utils.constants import DEFAULT_FFT_CONV1_KERNEL_SIZE, DEFAULT_FFT_CONV2_KERNEL_SIZE, DEFAULT_FFT_LENGTH, DEFAULT_HOP_LENGTH, DEFAULT_NUM_DECODER_LAYERS, DEFAULT_NUM_ENCODER_LAYERS, DEFAULT_NUM_HEADS, DEFAULT_DIM_HEAD, DEFAULT_DIM_FEEDFORWARD, DEFAULT_NUM_MELS, DEFAULT_NUM_POSTNET_LAYERS, DEFAULT_POSTNET_HIDDEN_DIM, DEFAULT_POSTNET_KERNEL_SIZE, DEFAULT_PREDICTOR_KERNEL_SIZE, DEFAULT_SAMPLE_RATE, DEFAULT_WIN_LENGTH, DEFAULT_VARIANCE_BINS
from utils.model import FastSpeech2, FastSpeech2Config


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
    parser.add_argument("-t", "--tags-file", type=pathlib.Path, help="包含标签的文件路径，格式是每行一个标签，默认不使用标签")
    parser.add_argument("-nm", "--num-mels", type=int, default=DEFAULT_NUM_MELS, help="梅尔频率倒谱系数(MFCC)的数量，默认为 %(default)s")
    parser.add_argument("-sr", "--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE, help="目标采样率(Hz)，默认为 %(default)s")
    parser.add_argument("-fl", "--fft-length", type=int, default=DEFAULT_FFT_LENGTH, help="FFT窗口长度，默认为 %(default)s")
    parser.add_argument("-wl", "--win-length", type=int, default=DEFAULT_WIN_LENGTH, help="窗函数长度，默认为 %(default)s")
    parser.add_argument("-hl", "--hop-length", type=int, default=DEFAULT_HOP_LENGTH, help="帧移长度，默认为 %(default)s")
    parser.add_argument("-nh", "--num-heads", type=int, default=DEFAULT_NUM_HEADS, help="注意力头的数量，默认为 %(default)s")
    parser.add_argument("-dh", "--dim-head", type=int, default=DEFAULT_DIM_HEAD, help="每个注意力头的维度，默认为 %(default)s")
    parser.add_argument("-df", "--dim-feedforward", type=int, default=DEFAULT_DIM_FEEDFORWARD, help="前馈网络的隐藏层维度，默认为 %(default)s")
    parser.add_argument("-fk1", "--fft-kernel-size-1", type=int, default=DEFAULT_FFT_CONV1_KERNEL_SIZE, help="FFT 第一个卷积核大小，默认为 %(default)s")
    parser.add_argument("-fk2", "--fft-kernel-size-2", type=int, default=DEFAULT_FFT_CONV2_KERNEL_SIZE, help="FFT 第二个卷积核大小，默认为 %(default)s")
    parser.add_argument("-prk", "--predictor-kernel-size", type=int, default=DEFAULT_PREDICTOR_KERNEL_SIZE, help="预测器卷积核大小，默认为 %(default)s")
    parser.add_argument("-vb", "--variance-bins", type=int, default=DEFAULT_VARIANCE_BINS, help="变异性预测器的 bins 数量，默认为 %(default)s")
    parser.add_argument("-dp", "--postnet-hidden-dim", type=int, default=DEFAULT_POSTNET_HIDDEN_DIM, help="后处理网络的隐藏层维度，默认为 %(default)s")
    parser.add_argument("-pok", "--postnet-kernel-size", type=int, default=DEFAULT_POSTNET_KERNEL_SIZE, help="后处理网络中卷积层的卷积核大小，默认为 %(default)s")
    parser.add_argument("-el", "--num-encoder-layers", type=int, default=DEFAULT_NUM_ENCODER_LAYERS, help="编码器层数，默认为 %(default)s")
    parser.add_argument("-dl", "--num-decoder-layers", type=int, default=DEFAULT_NUM_DECODER_LAYERS, help="解码器层数，默认为 %(default)s")
    parser.add_argument("-pl", "--num-postnet-layers", type=int, default=DEFAULT_NUM_POSTNET_LAYERS, help="后处理网络中卷积层的数量，默认为 %(default)s")
    parser.add_argument("-u", "--seed", default=8964, type=int, help="初始化检查点的种子，保证训练过程可复现，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 检查 FFT 窗口长度是否为偶数
    if args.fft_length % 2 == 1:
        raise RuntimeError("FFT 窗口长度必须为偶数。")
    # 检查注意力头的维度是否为偶数
    if args.dim_head % 2 == 1:
        raise RuntimeError("注意力头的维度必须为偶数。")

    # 加载分词器
    if not (tokenizer_path := args.ckpt_path / "tokenizer").exists():
        raise RuntimeError("你应该先训练分词器再初始化检查点，因为创建模型模型需要提供分词器的大小。")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # 检查标签文件是否存在
    if args.tags_file:
        tags = [tag.strip() for tag in args.tags_file.read_text("utf-8").splitlines() if tag.strip()]
    else:
        # 如果没有自定义标签，则警告
        warnings.warn(
            "你没有提供标签，这会大大削弱模型的表现力。\n"
            "如果你不确定训练时数据集是否带有标签，最保险的方法，\n"
            "请运行`python list_tags_from_datasets.py /path/to/dataset/metadata.json -o /path/to/tags.txt`，\n"
            "然后在运行本脚本时带上参数`-t /path/to/tags.txt`"
        )
        tags = ["default"]

    # 创建标签编码器
    tag_label = {tag: idx for idx, tag in enumerate(tags)}
    tag_label_encoder = TagLabelEncoder(tag_label)

    # 设置随机种子，确保可复现性
    set_seed(args.seed)

    # 初始化模型
    model = FastSpeech2(FastSpeech2Config(
        vocab_size=len(tokenizer),
        num_tags=len(tag_label),
        num_mels=args.num_mels,
        num_heads=args.num_heads,
        dim_head=args.dim_head,
        dim_feedforward=args.dim_feedforward,
        fft_conv1_kernel_size=args.fft_kernel_size_1,
        fft_conv2_kernel_size=args.fft_kernel_size_2,
        predictor_kernel_size=args.predictor_kernel_size,
        variance_bins=args.variance_bins,
        postnet_hidden_dim=args.postnet_hidden_dim,
        postnet_kernel_size=args.postnet_kernel_size,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        num_postnet_layers=args.num_postnet_layers,
    ))

    # 初始化优化器
    optimizer = optim.AdamW(model.parameters())

    # 生成模型生成配置
    generation_config = {
        "num_heads": args.num_heads,
        "sample_rate": args.sample_rate,
        "fft_length": args.fft_length,
        "hop_length": args.hop_length,
        "win_length": args.win_length,
    }

    # 创建 SummaryWriter，可视化模型初始化状态
    shutil.rmtree(args.ckpt_path / "logdir", ignore_errors=True)
    writer = SummaryWriter(args.ckpt_path / "logdir/default")
    writer.add_embedding(model.embedding.weight, [token.replace("\n", "[NEWLINE]").replace(" ", "[SPACE]") for token in tokenizer.convert_ids_to_tokens(range(len(tokenizer)))], tag="Init/Text Embedding")
    writer.add_embedding(model.tag_embedding.weight, [tag_label_encoder.id_to_tag[token_id] for token_id in range(len(tag_label_encoder))], tag="Init/Tag Embedding")
    writer.close()

    # 保存为检查点
    save_checkpoint(args.ckpt_path, model.state_dict(), optimizer.state_dict(), 0, generation_config, tag_label_encoder)

    # 打印初始化成功信息
    print(f"检查点初始化成功，已保存到 {args.ckpt_path}")


if __name__ == "__main__":
    main(parse_args())
