# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
from typing import Optional
import tgt
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
    parser = argparse.ArgumentParser(description="从 MFA 生成的 TextGrid 文件中提取音素时长信息，并将其注入到数据集元数据中")
    parser.add_argument("dataset", type=pathlib.Path, help="数据集元数据文件的路径")
    parser.add_argument("output", type=pathlib.Path, help="增强后的数据集元数据文件的输出路径")
    parser.add_argument("-c", "--no-clean", action="store_true", help="不删除中间处理文件（TextGrid 和 .lab 文件）")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 读取并解析元数据
    metadata = orjson.loads(args.dataset.read_bytes())

    # 遍历元数据
    for audio_path, audio_metadata in tqdm(metadata.items()):
        # 构建 TextGrid 和语料文件路径（替换音频扩展名为`.TextGrid`）
        textgrid_path = (args.dataset.parent / audio_path).with_suffix(".TextGrid")
        corups_path = textgrid_path.with_suffix(".lab")

        # 跳过缺失 TextGrid 的文件
        if not textgrid_path.exists():
            continue

        # 读取 TextGrid 文件并提取每个音素的开始时间、结束时间和文本内容
        phones_info = []
        for phone in tgt.io.read_textgrid(textgrid_path).get_tier_by_name("phones"):
            phones_info.append((float(phone.start_time), float(phone.end_time), phone.text))

        # 将音素信息注入到音频元数据中
        audio_metadata["phones"] = phones_info

        # 如果启用清理选项，删除中间处理文件
        if not args.no_clean:
            textgrid_path.unlink(missing_ok=True)
            corups_path.unlink(missing_ok=True)

    # 将增强后的元数据写入输出文件
    args.output.write_bytes(orjson.dumps(metadata))


if __name__ == "__main__":
    main(parse_args())
