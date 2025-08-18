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
    parser = argparse.ArgumentParser(description="应用互斥标签，将未出现在正面标签中的同组互斥标签加入负面标签")
    parser.add_argument("input_dir", type=pathlib.Path, help="包含待处理 JSON 文件的输入目录路径")
    parser.add_argument("output_dir", type=pathlib.Path, help="处理结果输出的目录路径")
    parser.add_argument("mutually_exclusive_groups", type=pathlib.Path, help="定义互斥标签组的 JSON 文件路径")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 获取数据集文件列表
    files = [
        file
        for file in args.input_dir.rglob("*.*")
        if file.suffix.lower() == ".json"
    ]

    # 加载互斥标签组并转换为集合列表
    mutually_exclusive_groups = orjson.loads(args.mutually_exclusive_groups.read_bytes())
    mutually_exclusive_groups = [set(group) for group in mutually_exclusive_groups]

    # 遍历数据集每一个元数据文件
    for file in tqdm(files):
        # 加载元数据
        metadata = orjson.loads(file.read_bytes())

        # 获取负面提示并转化为集合
        negative_prompt = set(metadata.get("negative_prompt", []))

        # 如果存在正面提示，则遍历每一个标签互斥组
        if positive_prompt := metadata.get("positive_prompt"):
            positive_prompt = set(positive_prompt)
            for group in mutually_exclusive_groups:
                # 如果正面标签有且仅有一个标签在互斥标签组内，那么把这个组的其他标签全部加入负面提示
                # 如果用户同时在正面提示指定了一个以上的同一个互斥组的标签，那么可能是用户自有需求，不要动它
                if len(intersection := group & positive_prompt) == 1:
                    negative_prompt.update(group - intersection)
        
        # orjson 不支持序列化集合，所以需要先把它转化为列表
        metadata["negative_prompt"] = list(negative_prompt)

        # 构造输出路径，并创建输出目录
        output_path = args.output_dir / file.relative_to(args.input_dir)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 写入文件
        output_path.write_bytes(orjson.dumps(metadata))


if __name__ == "__main__":
    main(parse_args())
