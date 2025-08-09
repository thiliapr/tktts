# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
from collections.abc import Callable, Iterable
from typing import Optional, TypeVar

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
        return func(*args[0])

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
    import torch
    import gc

    # 执行 Python 垃圾回收
    gc.collect()

    # 检查是否有可用的 CUDA 设备
    if torch.cuda.is_available():
        # 仅在 CUDA 设备上调用 empty_cache()
        torch.cuda.empty_cache()
