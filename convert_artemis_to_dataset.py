# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import pathlib
import shutil
from typing import Optional
import orjson
from tqdm import tqdm
from utils.dataset import AudioMetadata


def convert_and_save(
    script_files: list[pathlib.Path],
    character_label_mapping: dict[str, list[str]],
    output_dir: pathlib.Path,
    global_positive_prompt: list[str],
    global_negative_prompt: list[str]
) -> dict[str, AudioMetadata]:
    """
    将脚本文件中的对话转换并保存到指定目录。

    该函数处理每个对话，创建角色输出目录，复制音频文件，并生成相应的标签文件。

    Args:
        script_files: 包含脚本文件路径的列表。
        character_label_mapping: 角色标签映射表，包含角色ID与其标签的对应关系。
        output_dir: 输出目录，用于保存转换后的文件。
        global_positive_prompt: 额外的正面标签列表，将与角色特定标签合并。
        global_negative_prompt: 额外的负面标签列表，将与角色特定标签合并。
    """
    # 初始化进度条、元数据
    progress_bar = tqdm()
    dialogue_lengths = []
    metadata = {}

    for script_file_idx, script_file in enumerate(script_files):
        # 读取脚本文件内容
        dialogues = orjson.loads(script_file.read_bytes())

        # 更新进度条总数，计算方式: 当前对话总数 + 预计剩余对话数
        # 具体来说，预计剩余对话数 = (当前对话平均长度) * (剩余脚本文件数)
        dialogue_lengths.append(len(dialogues))
        progress_bar.total = int(sum(dialogue_lengths) + (sum(dialogue_lengths) / len(dialogue_lengths) * (len(script_files) - script_file_idx - 1)))

        # 更新进度条描述
        progress_bar.set_description(f"{script_file_idx + 1}/{len(script_files)}")

        # 处理每个对话
        for dialogue in dialogues:
            # 创建角色输出目录
            character_dir = output_dir / dialogue["character_id"]
            character_dir.mkdir(parents=True, exist_ok=True)

            # 获取音频文件路径和名称
            target_audio_path = character_dir / pathlib.Path(dialogue["path"]).name

            # 复制到目标路径
            try:
                shutil.copy(dialogue["path"], target_audio_path)
            except OSError as e:
                print(f"在将 `{dialogue['path']}` 复制到 `{target_audio_path}` 时发生了错误: {e}")
                target_audio_path.unlink(missing_ok=True)
                continue

            # 生成标签列表
            positive_prompt = global_positive_prompt.copy()
            negative_prompt = global_negative_prompt.copy()

            # 从映射表获取角色特定标签
            character_label = character_label_mapping.get(dialogue["character_id"], {})
            positive_prompt += character_label.get("positive_prompt", [])
            negative_prompt += character_label.get("negative_prompt", [])

            # 添加到元数据
            metadata[target_audio_path.relative_to(output_dir).as_posix()] = {
                "text": dialogue["text_content"],
                "positive_prompt": positive_prompt,
                "negative_prompt": negative_prompt
            }

            # 更新进度条
            progress_bar.update()
    
    return metadata


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="将 Artemis 引擎游戏的语音对话转换为 TkTTS 数据集")
    parser.add_argument("script_dir", type=pathlib.Path, help="包含预处理后脚本文件的目录，这些文件应是由 extract_artemis.py 生成的 JSON 文件")
    parser.add_argument("character_label_mapping", type=pathlib.Path, help="格式为 {角色ID: [标签1, 标签2, ...]}，需要手动创建此映射")
    parser.add_argument("output_dir", type=pathlib.Path, help="结构为 {输出目录}/{角色ID}/{音频文件} 和对应的 JSON 标签文件")
    parser.add_argument("-p", "--global-positive-prompt", type=str, action="append", default=[], help="给提取出来的数据集加全局正面标签，可以多次使用此选项。例如: 用`-p source:セレクトオブリージュ`表示数据集来源")
    parser.add_argument("-n", "--global-negative-prompt", type=str, action="append", default=[], help="给提取出来的数据集加全局负面标签，可以多次使用此选项")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 获取脚本文件
    script_files = list(args.script_dir.glob("*.json"))

    # 获取角色标签映射表
    character_label_mapping = orjson.loads(args.character_label_mapping.read_bytes())

    # 执行转换任务
    metadata = convert_and_save(script_files, character_label_mapping, args.output_dir, args.global_positive_prompt, args.global_negative_prompt)    

    # 将元数据写入一个文件，方便查找
    (args.output_dir / "metadata.json").write_bytes(orjson.dumps(metadata))


if __name__ == "__main__":
    main(parse_args())
