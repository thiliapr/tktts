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
    parser.add_argument("input_metadata", type=pathlib.Path, help="待处理元数据文件")
    parser.add_argument("output_filename", type=str, help="结果输出的元数据文件名。结果元数据文件将与待处理元数据文件同一目录")
    parser.add_argument("mutually_exclusive_groups", type=pathlib.Path, help="定义互斥标签组的 JSON 文件路径")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 加载互斥标签组并转换为集合列表
    mutually_exclusive_groups = orjson.loads(args.mutually_exclusive_groups.read_bytes())
    mutually_exclusive_groups = [set(group) for group in mutually_exclusive_groups]

    # 遍历数据集每一个音频元数据
    metadata = orjson.loads(args.input_metadata.read_bytes())
    for audio_path, audio_metadata in tqdm(metadata.items()):
        # 获取负面提示并转化为集合
        negative_prompt = set(audio_metadata.get("negative_prompt", []))

        # 如果存在正面提示，则遍历每一个标签互斥组
        if positive_prompt := audio_metadata.get("positive_prompt"):
            positive_prompt = set(positive_prompt)
            for group in mutually_exclusive_groups:
                # 如果正面标签有且仅有一个标签在互斥标签组内，那么把这个组的其他标签全部加入负面提示
                # 如果用户同时在正面提示指定了一个以上的同一个互斥组的标签，那么可能是用户自有需求，不要动它
                if len(intersection := group & positive_prompt) == 1:
                    negative_prompt.update(group - intersection)
        
        # orjson 不支持序列化集合，所以需要先把它转化为列表
        metadata[audio_path]["negative_prompt"] = list(negative_prompt)

    # 写入输出
    (args.input_metadata.parent / args.output_filename).write_bytes(orjson.dumps(metadata))

    # 打印提示增强成功信息
    print(f"提示词增强成功，增强后的元数据保存在 {args.output_filename}")


if __name__ == "__main__":
    main(parse_args())
