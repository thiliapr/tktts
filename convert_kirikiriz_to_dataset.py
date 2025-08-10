# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import multiprocessing
import pathlib
import shutil
from typing import Optional
import orjson
from tqdm import tqdm
from utils.tookit import parallel_map


def convert_and_save(
    rank: int,
    script_files: list[pathlib.Path],
    character_label_mapping: dict[str, list[str]],
    output_dir: pathlib.Path,
    extra_tags: list[str]
):
    """
    将脚本文件中的对话转换并保存到指定目录。

    该函数处理每个对话，创建角色输出目录，复制音频文件，并生成相应的标签文件。

    Args:
        rank: 当前工作进程的排名，用于控制进度条的显示。
        script_files: 包含脚本文件路径的列表。
        character_label_mapping: 角色标签映射表，包含角色ID与其标签的对应关系。
        output_dir: 输出目录，用于保存转换后的文件。
        extra_tags: 额外的标签列表，将与角色特定标签合并。
    """
    for script_file in tqdm(script_files, disable=rank != 0):
        # 处理每个对话
        for dialogue in orjson.loads(script_file.read_bytes()):
            # 解包，格式: (角色名, 文本内容, 音频文件路径, 是否循环语音)
            character_name, text_content, audio_file, is_loop_voice = dialogue

            # 获取音频文件路径和名称
            target_audio_path = output_dir / pathlib.Path(audio_file).name

            # 复制到目标路径
            try:
                shutil.copy(audio_file, target_audio_path)
            except OSError as e:
                print(f"在将 `{audio_file}` 复制到 `{target_audio_path}` 时发生了错误: {e}")
                target_audio_path.unlink(missing_ok=True)
                continue

            # 生成标签列表
            tags = extra_tags.copy()

            # 如果是循环语音，那么加上`sound:loop`标签
            if is_loop_voice:
                tags.append("sound:loop")

            # 从映射表获取角色特定标签
            tags += character_label_mapping[character_name]

            # 写入标签文件
            audio_metadata_file = (target_audio_path.parent / (target_audio_path.name + ".json"))
            audio_metadata_file.write_bytes(orjson.dumps({
                "text": text_content,
                "tags": tags
            }))


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="将 Kirikiri Z 引擎游戏的语音对话转换为 TkTTS 数据集")
    parser.add_argument("script_dir", type=pathlib.Path, help="包含预处理后脚本文件的目录，这些文件应是由 extract_kirikiriz.py 生成的 JSON 文件")
    parser.add_argument("character_label_mapping", type=pathlib.Path, help="格式为 {角色ID: [标签1, 标签2, ...]}，需要手动创建此映射")
    parser.add_argument("output_dir", type=pathlib.Path, help="结构为 {输出目录}/{角色ID}/{音频文件} 和对应的 JSON 标签文件")
    parser.add_argument("-t", "--tag", type=str, action="append", default=[], help="给提取出来的数据集加全局标签，可以多次使用此选项。例如: 用`-t source:千恋＊万花`表示数据集来源")
    parser.add_argument("-n", "--num-workers", type=int, default=multiprocessing.cpu_count(), help="执行任务的进程数，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 创建输出目录
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 获取脚本文件
    script_files = list(args.script_dir.glob("*.json"))

    # 获取角色标签映射表
    character_label_mapping = orjson.loads(args.character_label_mapping.read_bytes())

    # 并行执行转换任务
    parallel_map(convert_and_save, [
        (worker_id, script_files[worker_id::args.num_workers], character_label_mapping, args.output_dir, args.tag)
        for worker_id in range(args.num_workers)
    ])


if __name__ == "__main__":
    main(parse_args())
