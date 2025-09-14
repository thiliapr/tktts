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
    parser = argparse.ArgumentParser(description="复制音频到指定目录，并从数据集元数据提取出文本，保存到`audio_path.with_suffix('.lab')`，方便 Montreal Forced Aligner 对齐")
    parser.add_argument("dataset", type=pathlib.Path, help="数据集元数据文件的路径，文本将会保存到该元数据文件所处的目录下")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 读取并解析元数据
    metadata = orjson.loads(args.dataset.read_bytes())

    # 检测是否存在同一个文件夹内、同名不同后缀的音频文件
    corups_paths = set()
    for audio_path in metadata.keys():
        corpus_path = pathlib.PurePath(audio_path).with_suffix(".lab")
        if corups_paths:
            raise Exception(f"文件夹 {corpus_path.parent} 中存在至少两个同名但后缀不同的音频文件，它们的文件名都是 {corpus_path.stem}。由于 Montreal Forced Aligner 要求同一名称只能对应一个音频文件（即使后缀不同），当前的文件冲突会导致程序无法正常处理")
        corups_paths.add(corpus_path)

    # 遍历元数据
    for audio_path, audio_metadata in tqdm(metadata.items()):
        corpus_path = (args.dataset.parent / audio_path).with_suffix(".lab")

        # 提取文本并写入 .lab 文件
        corpus_path.write_text(audio_metadata["text"], encoding="utf-8")


if __name__ == "__main__":
    main(parse_args())
