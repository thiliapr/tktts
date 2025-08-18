# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
from typing import Optional
from utils.checkpoint import load_checkpoint


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="输出检查点支持的所有标签")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点加载路径")
    parser.add_argument("-o", "--output-file", type=str, default="-", help="输出路径。如果指定为`-`，则输出路径为标准输出。默认为输出到标准输出。")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 获取输出函数
    if args.output_file == "-":
        output_func = lambda x: print(x)
    else:
        output_file = pathlib.Path(args.output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        output_func = lambda x: output_file.write_text(x, encoding="utf-8")

    # 加载检查点
    _, _, _, _, tag_label_encoder = load_checkpoint(args.ckpt_path)

    # 获取所有标签并排序
    tags = sorted(tag_label_encoder.vocab.keys())
    tags.remove("[PAD]")  # 移除填充标签

    # 打印标签列表
    output_func("\n".join(tags))


if __name__ == "__main__":
    main(parse_args())
