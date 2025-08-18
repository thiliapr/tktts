# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # 禁用tokenizers的并行处理，避免冲突

import random
import pathlib
import argparse
import multiprocessing
from typing import Any, Optional, Union
import orjson
from tqdm import tqdm
from tokenizers import Tokenizer, models, trainers
from transformers import PreTrainedTokenizerFast
from utils.tookit import parallel_map


def train_tokenizer(model_data_samples: list[str], vocab_size: int, min_frequency: int):
    """
    训练专门用于处理模型数据的 tokenizer

    Args:
        model_data_samples: 多个模型数据样本
        vocab_size: 词汇表大小，控制分词器的容量
        min_frequency: 最小出现频率，低于此值的token将被忽略

    Returns:
        训练好的 tokenizer 实例
    """
    # 初始化BPE分词器
    tokenizer = Tokenizer(models.BPE())

    # 准备训练器配置
    trainer = trainers.BpeTrainer(
        special_tokens=["[UNK]"],
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        show_progress=False
    )

    # 使用样本数据训练分词器
    tokenizer.train_from_iterator(model_data_samples, trainer=trainer)

    # 包装
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
    )

    return wrapped_tokenizer


def get_samples_worker(rank: int, metadata_files: list[pathlib.Path]) -> list[str]:
    """
    从元数据中提取标签和文本

    Args:
        rank: 进程编号，用于进度条显示
        metadata_files: 元数据文件的路径列表

    Returns:
        文本列表
    """
    # 初始化文本列表
    texts = []

    # 遍历每个元数据文件
    for metadata_file in tqdm(metadata_files, disable=rank != 0):
        # 跳过没有对应组合频谱的元数据
        if not metadata_file.with_suffix("").exists():
            continue

        # 读取元数据
        metadata = orjson.loads(metadata_file.read_bytes())

        # 添加标签和文本
        texts.append(metadata["text"])

    return texts


def get_samples(metadata_dirs: list[pathlib.Path], num_samples: int, num_workers: int) -> tuple[list[str], set[str]]:
    """
    从指定目录中提取元数据的标签和文本

    Args:
        dirs: 包含 JSON 元数据文件的目录列表
        num_samples: 随机抽选多少个样本
        num_workers: 执行任务的进程数

    Returns:
        包含文本列表和标签集合的元组列表
    """
    # 获取样本文件
    metadata_files = [
        metadata_file
        for metadata_dir in metadata_dirs
        for metadata_file in metadata_dir.rglob("*.*")
        if metadata_file.suffix.lower() == ".json"
    ]

    # 如果样本数量超过指定数量，则随机抽样
    if len(metadata_files) > num_samples:
        print(f"样本数量超过 {num_samples}，随机抽样...")
        metadata_files = random.sample(metadata_files, num_samples)

    # 处理所有样本文件
    results = parallel_map(get_samples_worker, [(worker_id, metadata_files[worker_id::num_workers]) for worker_id in range(num_workers)])

    # 提取文本
    samples = [text for worker_texts in results for text in worker_texts]

    return samples


def validate(samples: list[str], tokenizer: PreTrainedTokenizerFast) -> dict[str, Union[int, float]]:
    """
    评估分词器的效果

    Args:
        samples: 样本列表
        tokenizer: 要评估的分词器

    Returns:
        字典，包含:
        - 总字符数量
        - 平均序列长度占字符总数的百分比
        - 使用的词汇占总词汇表的比例
    """
    # 初始化统计变量
    total_chars = 0
    total_seq_length = 0
    words_used_set = set()

    # 遍历样本进行评估
    for data in tqdm(samples, desc="评估效果"):
        total_chars += len(data)
        encoded = tokenizer.encode(data)
        total_seq_length += len(encoded)
        words_used_set.update(encoded)

    # 统计
    avg_seq_ratio = total_seq_length / total_chars if total_chars > 0 else 0
    vocab_usage = len(words_used_set) / len(tokenizer)

    return {"total_chars": total_chars, "avg_seq_ratio": avg_seq_ratio, "vocab_usage": vocab_usage}


def print_validation_results(metrics: dict[str, Any]):
    """
    以易读格式打印验证结果

    参数:
        metrics: validate函数返回的评估指标字典
    """
    print(f"- 总字符数: {metrics['total_chars']}")
    print(f"- 平均序列长度/字符数: {metrics['avg_seq_ratio']:.2%}")
    print(f"- 词汇表使用率: {metrics['vocab_usage']:.2%}")


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="从游戏脚本提取每个对话的语音和文本。本脚本仅适用于 Artemis 引擎游戏。")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="分词器保存路径，将创建tokenizer子目录")
    parser.add_argument("-t", "--train-samples", type=pathlib.Path, action="append", required=True, help="训练集目录路径，包含MIDI样本文件")
    parser.add_argument("-v", "--valid-samples", type=pathlib.Path, action="append", help="验证集目录路径，包含MIDI样本文件")
    parser.add_argument("-nt", "--train-samples-count", type=int, default=2 ** 15, help="训练集样本数量，默认为 %(default)s")
    parser.add_argument("-nv", "--valid-samples-count", type=int, default=2 ** 15, help="验证集样本数量，默认为 %(default)s")
    parser.add_argument("-s", "--vocab-size", type=int, default=10000, help="分词器词汇表大小，默认为 %(default)s")
    parser.add_argument("-f", "--min-frequency", type=int, default=24, help="token最小出现频率阈值，默认为 %(default)s")
    parser.add_argument("-nw", "--num-workers", type=int, default=multiprocessing.cpu_count(), help="执行任务的进程数，默认为 %(default)s")
    parser.add_argument("-y", "--force", action="store_true", help="即使检查点已经存在分词器也要训练新的分词器")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    tokenizer_path = args.ckpt_path / "tokenizer"

    # 如果检查点已有分词器，且未指定重新训练分词器，则跳过
    if tokenizer_path.exists() and not args.force:
        print("已存在分词器，跳过训练。如果想重新训练分词器，请在参数指定`-y/--force`")
        return

    # 检查并创建检查点目录
    args.ckpt_path.mkdir(parents=True, exist_ok=True)

    # 处理所有训练样本文件
    train_samples = get_samples(args.train_samples, args.train_samples_count, args.num_workers)

    # 训练分词器
    print("开始训练分词器...")
    tokenizer = train_tokenizer(
        train_samples,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency
    )

    # 保存分词器
    tokenizer.save_pretrained(tokenizer_path)
    print(f"分词器已保存到 {tokenizer_path}")

    # 输出基本信息
    print("\n训练结果:")
    print(f"- 词汇表大小: {tokenizer.vocab_size}")
    print(f"- 训练样本数: {len(train_samples)}")

    # 评估训练集效果
    print("\n训练集评估:")
    train_metrics = validate(train_samples, tokenizer)
    print_validation_results(train_metrics)

    # 如果有验证集，评估验证集效果
    if args.valid_samples:
        print("\n验证集评估:")

        # 处理所有验证样本文件
        valid_samples = get_samples(args.valid_samples, args.valid_samples_count, args.num_workers)

        # 评估验证集效果
        valid_metrics = validate(valid_samples, tokenizer)
        print_validation_results(valid_metrics)


if __name__ == "__main__":
    main(parse_args())
