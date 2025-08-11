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
    parser = argparse.ArgumentParser(description="输出所有检查点支持的标签")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点加载路径")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 加载检查点
    tokenizer, _, _, _ = load_checkpoint(args.ckpt_path)

    # 获取所有标签并排序（以`[`开头并以`]`结尾的 special_token）
    tags = sorted(token[1:-1] for token in tokenizer.get_added_vocab() if token.startswith("[") and token.endswith("]"))

    # 删除特殊 token
    tags.remove("UNK")

    # 打印标签列表
    for tag in tags:
        print(tag)


if __name__ == "__main__":
    main(parse_args())
