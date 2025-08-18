# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
from typing import Optional
import orjson
from tqdm import tqdm


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="获取数据集出现的标签")
    parser.add_argument("datasets", nargs="+", type=pathlib.Path, help="数据集路径列表")
    parser.add_argument("-o", "--output-file", type=str, default="-", help="输出路径。如果指定为`-`，则输出路径为标准输出。默认为输出到标准输出。")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 获取输出函数
    if args.output_file == "-":
        output_func = lambda x: print(x)
    else:
        output_file = pathlib.Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_func = lambda x: output_file.write_text(x + "\n", encoding="utf-8")

    # 获取数据集文件列表
    files = [
        file
        for dataset_dir in args.datasets
        for file in dataset_dir.rglob("*.*")
        if file.suffix.lower() == ".json"
    ]

    # 加载数据集并获取标签
    tags = set()
    progress_bar = tqdm(files)
    for file in progress_bar:
        tags |= set(orjson.loads(file.read_bytes()).get("positive_prompt", []))
        tags |= set(orjson.loads(file.read_bytes()).get("negative_prompt", []))
        progress_bar.set_postfix(found=len(tags))
    
    # 按标签名称排序
    tags = sorted(tags)

    # 打印标签
    for tag in tags:
        output_func(tag)


if __name__ == "__main__":
    main(parse_args())
