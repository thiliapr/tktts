# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
import argparse
import multiprocessing
from typing import Optional
import orjson
from tqdm import tqdm
from utils.tookit import clean_text, find_highest_priority_file, parallel_map
from utils.constants import AUDIO_SUFFIXES


def convert_and_save(
    rank: int,
    script_files: list[pathlib.Path],
    voice_dir: pathlib.Path,
    output_dir: pathlib.Path,
    audio_suffixes: list[str],
) -> set[str]:
    character_names = set()

    # 遍历所有脚本文件（主进程显示进度条）
    for script_path in tqdm(script_files, disable=rank != 0):
        # 读取并解析脚本文件
        script = orjson.loads(script_path.read_bytes())

        # 初始化文本和音频映射列表
        # 格式: (角色名, 文本内容, 音频文件路径, 是否循环语音)
        text_audio_mappings: list[tuple[str, str, str, bool]] = []
        loop_voice_files = set()

        for scene in script.get("scenes", []):
            for text in scene.get("texts", []):
                # 提取文本内容并清理
                text_content = clean_text(text[2])

                # 检查文本是否包含语音信息
                if voice := text[3]:
                    # 提取语音信息
                    voice = voice[0]
                    voice_file = voice["voice"]

                    # 查找最佳匹配的语音文件
                    audio_file = find_highest_priority_file(voice_dir.glob(f"{voice_file}.*"), audio_suffixes)

                    # 找不到文件就跳过
                    if not audio_file:
                        continue

                    # 添加到映射列表
                    character_names.add(voice["name"])
                    text_audio_mappings.append((voice["name"], text_content, str(audio_file), False))

                # 处理循环语音列表
                for voice in text[5].get("loopVoiceList", []):
                    voice_file = voice["voice"]

                    # 防止重复添加相同的循环语音文件
                    if voice_file in loop_voice_files:
                        continue

                    # 查找最佳匹配的语音文件
                    audio_file = find_highest_priority_file(voice_dir.glob(f"{voice_file}.*"), audio_suffixes)

                    # 找不到文件就跳过
                    if not audio_file:
                        continue

                    # 添加到映射列表
                    character_names.add(voice["name"])
                    loop_voice_files.add(voice_file)
                    text_audio_mappings.append((voice["name"], text_content, str(audio_file), True))

        (output_dir / script_path.with_suffix(".json").name).write_bytes(orjson.dumps(text_audio_mappings))

    return character_names


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="从游戏脚本提取每个对话的语音和文本。本脚本仅在 Kirikiki Z 引擎游戏《千恋 * 万花》测试。")
    parser.add_argument("script_dir", type=pathlib.Path, help="游戏经过 FreeMote.PsbDecompile 的 JSON 脚本目录")
    parser.add_argument("voice_dir", type=pathlib.Path, help="语音目录，包含语音文件（.ogg/.wav等）")
    parser.add_argument("output_dir", type=pathlib.Path, help="输出目录，解析后的数据集将存储在此处")
    parser.add_argument("-s", "--audio-suffix", type=str, action="append", default=[], help=f"指定优先音频后缀，可添加多个。内置音频后缀: {AUDIO_SUFFIXES}")
    parser.add_argument("-n", "--num-workers", type=int, default=multiprocessing.cpu_count(), help="执行任务的进程数，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 构建音频后缀名列表，按照优先级排序
    audio_suffixes = args.audio_suffix + AUDIO_SUFFIXES

    # 获取脚本文件路径
    files = list(args.script_dir.rglob("*.json"))

    # 执行任务
    results = parallel_map(convert_and_save, [
        (worker_id, files[worker_id::args.num_workers], args.voice_dir, args.output_dir, audio_suffixes)
        for worker_id in range(args.num_workers)
    ])

    # 合并所有进程的结果，结果是一个包含所有角色名的集合
    character_names = set(name for result in results for name in result)

    # 打印角色名称
    for name in character_names:
        print(name)


if __name__ == "__main__":
    main(parse_args())
