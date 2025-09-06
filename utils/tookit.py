# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import gc
import pathlib
from typing import Optional, TypeVar, Union
from collections.abc import Callable, Iterable, Sequence
import torch
import numpy as np

QUOTE_CHARS = "“”「」『』'\""
SPACE_CHARS = "\u3000"
T = TypeVar("T")


def find_highest_priority_file(files: Iterable[pathlib.Path], suffix_priorities: list[str]) -> Optional[pathlib.Path]:
    """
    查找并返回具有最高优先级后缀的文件
    
    工作流程:
    1. 过滤文件列表，保留后缀存在于优先级列表中的文件
    2. 若无可匹配文件，返回None
    3. 根据后缀在优先级列表中的位置进行排序（优先级列表中越靠前的后缀优先级越高）
    4. 返回优先级最高的文件

    Args:
        files: 待筛选的文件路径列表
        suffix_priorities: 后缀优先级列表，列表中位置越靠前的后缀优先级越高

    Returns:
        优先级最高的文件路径，若无匹配文件则返回None

    Examples:
        >>> files = [Path("audio.mp3"), Path("audio.wav"), Path("audio.ogg")]
        >>> find_highest_priority_file(files, [".wav", ".mp3", ".ogg"])
        Path('audio.wav')
        >>> find_highest_priority_file(files, [".ogg", ".mp3"])
        Path('audio.ogg')
    """
    # 筛选出后缀在优先级列表中的文件
    matching_files = [file for file in files if file.suffix in suffix_priorities]

    # 若无匹配文件，返回None
    if not matching_files:
        return None

    # 根据后缀在优先级列表中的索引排序（索引越小优先级越高）
    # 使用reverse=False（默认）确保按索引升序排列
    matching_files.sort(key=lambda file: suffix_priorities.index(file.suffix))

    # 返回优先级最高的文件
    return matching_files[0]


def clean_text(text: str) -> str:
    """
    清理文本内容，移除特殊空格和引号。

    Args:
        text: 待清理的文本内容

    Returns:
        清理后的文本内容

    Examples:
        >>> clean_text("  “Hello, World!”  ")
        'Hello, World!'
        >>> clean_text("「こんにちは」")
        'こんにちは'
    """
    # 替换所有特殊空格为普通空格
    for space_char in SPACE_CHARS:
        text = text.replace(space_char, " ")

    # 清理文本内容
    text = text.strip()

    # 移除首尾引号
    for quote_char in QUOTE_CHARS:
        text = text.removeprefix(quote_char).removesuffix(quote_char)

    return text


def parallel_map(func: Callable[..., T], args: list[tuple]) -> list[T]:
    """
    使用多进程并行执行函数。
    该函数将给定的函数应用于位置参数的每个元素，并根据位置参数的数量个的工作进程并行处理。

    Args:
        func: 要应用的函数，接受位置参数的元素作为参数
        args: 位置参数，包含要处理的数据

    Returns:
        包含函数应用结果的列表
    """
    # 单进程时直接执行函数而不创建子进程
    if len(args) == 1:
        return [func(*args[0])]

    from multiprocessing import Pool
    with Pool(processes=len(args)) as pool:
        return pool.starmap(func, args)


def empty_cache():
    """
    清理 CUDA 显存缓存并执行 Python 垃圾回收。

    本函数会先触发 Python 的垃圾回收机制，释放未被引用的内存。
    如果检测到有可用的 CUDA 设备，则进一步清理 CUDA 显存缓存，释放未被 PyTorch 占用但已缓存的 GPU 显存。

    Examples:
        >>> empty_cache()
    """
    # 执行 Python 垃圾回收
    gc.collect()

    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():
        # 仅在 CUDA 设备上调用 empty_cache()
        torch.cuda.empty_cache()

def convert_to_tensor(items: list[Union[torch.Tensor, np.ndarray, list[Union[float, int]]]]) -> list[torch.Tensor]:
    """
    将多种类型的数值列表统一转换为 PyTorch 张量列表

    该函数接受包含多种数值类型的列表，包括 PyTorch 张量、NumPy 数组和 Python 数值列表，
    并将它们统一转换为PyTorch张量格式。转换过程保持原始数据的数值精度和维度结构。

    Args:
        items: 输入数据列表，可包含 PyTorch 张量、NumPy 数组或数值列表

    Returns:
        转换后的 PyTorch 张量列表

    Examples:
        >>> convert_to_tensor([np.array([1, 2]), [3.0, 4.0], torch.tensor([5, 6])])
        [tensor([1, 2]), tensor([3., 4.]), tensor([5, 6])]
    """
    # 使用列表推导式批量转换所有输入项为PyTorch张量
    return [torch.tensor(item) for item in items]


def get_sequence_lengths(sequences: list[torch.Tensor]) -> torch.LongTensor:
    """
    计算每个序列张量的长度并返回长度张量

    遍历输入的序列张量列表，获取每个序列的第一维度大小（即序列长度），
    并将所有长度值组合成一个整型长张量返回。

    Args:
        sequences: 包含多个序列张量的列表

    Returns:
        包含各序列长度的整型张量

    Examples:
        >>> get_sequence_lengths([torch.rand(3, 5), torch.rand(2, 5)])
        tensor([3, 2])
    """
    # 使用列表推导式获取每个序列的长度，然后转换为张量
    return torch.tensor([len(sequence) for sequence in sequences], dtype=int)


def create_padding_mask(sequences: list[torch.Tensor]) -> torch.BoolTensor:
    """
    创建用于序列填充的布尔掩码张量

    根据输入序列的长度信息生成一个二维布尔掩码，其中 True 值表示对应位置为填充位置。
    掩码的维度为 [批次大小, 最大序列长度]，便于后续的填充操作和注意力机制计算。

    Args:
        sequences: 包含多个序列张量的列表

    Returns:
        二维布尔掩码张量，标识填充位置

    Examples:
        >>> create_padding_mask([torch.rand(3, 5), torch.rand(2, 5)])
        tensor([[False, False, False],
                [False, False,  True]])
    """
    # 获取各序列长度
    sequence_lengths = get_sequence_lengths(sequences)

    # 计算最大序列长度
    max_length = sequence_lengths.max().item()

    # 生成位置索引矩阵并与各序列长度比较，创建掩码
    position_indices = torch.arange(max_length, dtype=torch.long)

    # 通过广播机制比较位置索引与序列长度，生成填充掩码（True表示填充位置）
    return position_indices.unsqueeze(0) >= sequence_lengths.unsqueeze(1)
