# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
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
    parser = argparse.ArgumentParser(description="合并多个数据集的元数据文件")
    parser.add_argument("datasets", nargs="+", type=pathlib.Path, help="数据集元数据文件路径列表")
    parser.add_argument("-o", "--output-file", type=pathlib.Path, required=True, help="合并后的元数据输出路径，通常为`dataset/metadata.json`")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 验证输出目录是否包含所有输入数据集目录
    output_parent = args.output_file.parent
    if not all(metadata_file.parent.is_relative_to(output_parent) for metadata_file in args.datasets):
        raise RuntimeError(f"输出目录（{output_parent}）必须同时是所有输入数据集目录的父目录")

    # 合并元数据
    metadata = {
        (metadata_file.parent.relative_to(output_parent) / audio_path).as_posix(): audio_metadata
        for metadata_file in args.datasets
        for audio_path, audio_metadata in orjson.loads(metadata_file.read_bytes()).items()
    }

    # 将合并后的元数据序列化为 JSON 并写入输出文件
    args.output_file.write_bytes(orjson.dumps(metadata))


if __name__ == "__main__":
    main(parse_args())
