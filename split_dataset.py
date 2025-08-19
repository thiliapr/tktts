# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import random
from typing import Optional
import orjson


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="随机将一个数据集的元数据文件拆分为多个")
    parser.add_argument("dataset", type=pathlib.Path, help="数据集元数据文件路径")
    parser.add_argument("splits", nargs="+", type=str, help="输出文件名和拆分比例，格式为`filename:proportion`，如`train.json:9`和`val.json:1`")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 读取并解析元数据
    metadata = list(orjson.loads(args.dataset.read_bytes()).items())

    # 解析拆分配置
    splits = [str_split.split(":", 1) for str_split in args.splits]
    splits = [(filename, int(proportion)) for filename, proportion in splits]
    total_proportion = sum(proportion for _, proportion in splits)

    # 验证数据集大小是否满足拆分需求
    if len(metadata) < total_proportion:
        raise RuntimeError(f"音频数据数量（{len(metadata)}）小于指定的拆分比例总和（{total_proportion}）")

    # 随机打乱元数据
    random.shuffle(metadata)

    # 按比例拆分数据
    split_data = [metadata[rank::total_proportion] for rank in range(total_proportion)]

    # 写入拆分后的文件
    for filename, proportion in splits:
        # 从拆分数据中提取对应比例的数据
        subset = [item for chunk in split_data[:proportion] for item in chunk]
        split_data = split_data[proportion:]

        # 将子集写入对应文件
        (args.dataset.parent / filename).write_bytes(orjson.dumps(dict(subset)))


if __name__ == "__main__":
    main(parse_args())
