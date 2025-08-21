"提取 Artemis 引擎游戏每个对话的语音和文本。"

# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import argparse
import multiprocessing
import pathlib
from typing import Optional
from collections import Counter
import orjson
from tqdm import tqdm
from luaparser import ast as lua_ast
from utils.tookit import clean_text, find_highest_priority_file, parallel_map
from utils.constants import AUDIO_SUFFIXES


def is_field_key_matching(field: lua_ast.Field, target_key: str) -> bool:
    """
    判断Lua AST中的字段键是否与指定字符串匹配。

    工作流程：
    1. 检查字段的键是否为lua_ast.Name类型
    2. 若为Name类型，则进一步比较其标识符字符串是否与目标键相等
    3. 返回布尔值结果，表示是否匹配

    Args:
        field: Lua AST中的字段节点
        target_key: 用于比较的目标键字符串

    Returns:
        若字段键为Name类型且其标识符等于目标键，返回True；否则返回False

    Examples:
        >>> field = lua_ast.Field(key=lua_ast.Name(id='example'))
        >>> is_field_key_matching(field, 'example')
        True
        >>> is_field_key_matching(field, 'other')
        False
    """
    return isinstance(field.key, lua_ast.Name) and field.key.id == target_key


def extract_from_block(block: lua_ast.Field, lang_code: str) -> tuple[Optional[tuple[str, str, str]], str]:
    """
    从Lua AST字段块中提取语音信息和本地化文本内容

    主要处理游戏对话块中的语音角色信息和多语言文本内容。函数执行以下操作：
    1. 遍历字段块定位文本字段(text)
    2. 从语音事件(vo)中提取角色标识和语音文件
    3. 从指定语言字段中提取文本内容
    4. 处理特殊文本标记(如rt2换行标记)
    5. 清理文本中的特殊空格和引号字符

    处理流程详细说明：
    - 首先在字段块中查找文本字段(text)
    - 在文本字段内遍历所有事件(event)
    - 遇到语音事件(vo)时提取角色标识(ch)和语音文件(file)
    - 遇到目标语言事件时提取文本内容
    - 合并所有文本片段并清理特殊字符

    Args:
        block: 包含对话信息的Lua AST字段块
        lang_code: 目标语言代码(如"ja"), 应与语音语言一致

    Returns:
        - 语音信息元组(角色标识, 语音文件), 无语音时返回None
        - 清理后的文本内容字符串

    Examples:
        >>> voice_info, text = extract_voice_and_text_from_block(lua_block, "ja")
        >>> print(voice_info)
        ('角色标识', '角色名称', 'voice_file.ogg')
        >>> print(text)
        '清理后的日语文本'
    """
    character_id = None  # 语音角色ID
    character_name = None  # 语音角色名
    voice_file = None  # 语音文件路径
    text_segments = []  # 存储提取的文本片段

    # 遍历字段块中的所有字段
    for field in block.value.fields:
        # 跳过非文本字段
        if not is_field_key_matching(field, "text"):
            continue

        # 处理文本字段中的所有事件
        for event in field.value.fields:
            # 处理语音事件(vo)
            if is_field_key_matching(event, "vo"):
                for voice_item in event.value.fields:
                    # 提取语音角色和文件名
                    for voice_arg in voice_item.value.fields:
                        if is_field_key_matching(voice_arg, "file"):
                            voice_file = voice_arg.value.s
                        elif is_field_key_matching(voice_arg, "ch"):
                            character_id = voice_arg.value.s

            # 处理目标语言文本
            elif is_field_key_matching(event, lang_code):
                for text_item in event.value.fields:
                    for text_arg in text_item.value.fields:
                        # 处理普通字符串内容
                        if isinstance(text_arg.value, lua_ast.String):
                            text_segments.append(text_arg.value.s)

                        # 处理角色名字
                        elif is_field_key_matching(text_arg, "name"):
                            character_name = text_arg.value.fields[0].value.s

                        # 处理特殊换行标记(rt2)
                        elif isinstance(text_arg.value, lua_ast.Table) and isinstance(text_arg.value.fields[0].value, lua_ast.String) and text_arg.value.fields[0].value.s == "rt2":
                            text_segments.append("\n")

    # 构建语音信息元组(如果有语音文件)
    voice = (character_id, character_name, voice_file) if voice_file else None

    # 合并文本内容并清理
    content = clean_text("".join(text_segments))
    return voice, content


def convert_and_save(
    rank: int,
    script_files: list[pathlib.Path],
    voice_dir: pathlib.Path,
    output_dir: pathlib.Path,
    audio_suffixes: list[str],
    lang_code: str
) -> dict[str, Counter]:
    """
    将Lua脚本文件中的对话内容与对应的语音文件配对，生成JSON格式的映射文件。

    处理流程：
    1. 遍历所有Lua脚本文件
    2. 解析Lua AST结构并提取对话内容
    3. 提取对话中的角色信息和文本内容
    4. 查找匹配的语音文件（按扩展名优先级）
    5. 生成角色ID到角色名的频率统计
    6. 将文本-语音映射关系保存为JSON文件

    Args:
        process_rank: 当前处理进程的ID（用于控制进度条显示）
        script_files: 待处理的Lua脚本文件路径列表
        voice_directory: 语音文件存储的根目录
        output_directory: JSON输出文件目录
        audio_extensions: 支持的音频文件扩展名列表（按优先级排序）
        language_code: 目标语言代码（用于文本处理）

    Returns:
        角色ID到角色名称计数的映射字典

    Examples:
        >>> mapping = convert_and_save(
        ...     0,
        ...     [Path("script1.lua"), Path("script2.lua")],
        ...     Path("voices"),
        ...     Path("output"),
        ...     [".ogg", ".wav"],
        ...     "ja"
        ... )
        # 将在output目录生成script1.json和script2.json文件
        # 返回格式: {"char_001": Counter({"Alice": 5}), "char_002": Counter({"Bob": 3})}
    """
    # 角色ID到角色名称的计数映射
    character_counter = {}

    # 遍历所有脚本文件（主进程显示进度条）
    for script_path in tqdm(script_files, disable=rank != 0):
        # 读取并解析脚本文件的 Lua 表结构
        ast_content = script_path.read_text(encoding="utf-8")
        parsed_ast = lua_ast.parse(ast_content)

        # 获取 AST 中的对话
        text_blocks = parsed_ast.body.body[2].values[0].fields

        # 存储当前文件的文本-语音映射条目
        text_audio_mappings = []

        # 处理每个对话块
        for block in text_blocks:
            # 提取语音信息和对话文本
            voice_info, dialogue_text = extract_from_block(block, lang_code)

            # 跳过没有语音信息的对话
            if voice_info is None:
                continue

            # 解包语音信息
            character_id, character_name, voice_file = voice_info

            # 查找最佳匹配的语音文件
            audio_file = find_highest_priority_file((voice_dir / character_id).glob(f"{voice_file}.*"), audio_suffixes)

            # 找不到文件就跳过
            if not audio_file:
                continue

            # 添加文本-语音映射条目
            text_audio_mappings.append((character_id, dialogue_text, audio_file))

            # 更新角色名称计数器
            if character_id not in character_counter:
                character_counter[character_id] = Counter()
            character_counter[character_id][character_name] += 1

        # 写入JSON文件
        (output_dir / script_path.name).with_suffix(".json").write_bytes(orjson.dumps([
            {
                "path": str(file),
                "text_content": text_content,
                "character_id": character_id,
            }
            for character_id, text_content, file in text_audio_mappings
        ]))

    return character_counter


def parse_args(args: Optional[list[str]] = None) -> argparse.Namespace:
    """
    解析命令行参数。

    Args:
        args: 命令行参数列表。如果为 None，则使用 sys.argv。

    Returns:
        包含解析后的参数的命名空间对象。
    """
    parser = argparse.ArgumentParser(description="从游戏脚本提取每个对话的语音和文本。本脚本仅适用于 Artemis 引擎游戏。")
    parser.add_argument("script_dir", type=pathlib.Path, help="游戏的 AST 脚本目录")
    parser.add_argument("voice_dir", type=pathlib.Path, help="语音目录，包含语音文件（.ogg/.wav等）。提示: 语音文件通常储存在游戏根目录的`sound/vo`文件夹下")
    parser.add_argument("output_dir", type=pathlib.Path, help="输出目录，解析后的数据集将存储在此处")
    parser.add_argument("-l", "--lang-code", type=str, default="ja", help="语言代码，默认为 %(default)s")
    parser.add_argument("-s", "--audio-suffix", type=str, action="append", default=[], help=f"指定优先音频后缀，可添加多个。内置音频后缀: {AUDIO_SUFFIXES}")
    parser.add_argument("-n", "--num-workers", type=int, default=multiprocessing.cpu_count(), help="执行任务的进程数，默认为 %(default)s")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    """
    处理AST脚本文件并生成文本-语音配对JSON文件

    主要工作流程：
    1. 遍历输入目录中的所有AST脚本文件
    2. 解析每个脚本的Lua AST结构
    3. 提取对话中的语音信息和文本内容
    4. 查找对应的语音文件
    5. 生成包含文本-语音配对信息的JSON文件
    """
    # 确保输出目录存在
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 构建音频后缀名列表，按照优先级排序
    audio_suffixes = args.audio_suffix + AUDIO_SUFFIXES

    # 执行任务
    files = list(args.script_dir.rglob("*.ast"))
    results = parallel_map(convert_and_save, [
        (worker_id, files[worker_id::args.num_workers], args.voice_dir, args.output_dir, audio_suffixes, args.lang_code)
        for worker_id in range(args.num_workers)
    ])

    # 汇集计数器数据
    character_counter = {}
    for result in results:
        for key, counter in result.items():
            if key not in character_counter:
                character_counter[key] = Counter()
            character_counter[key] += counter

    # 打印每个标识对应的名称，以及每个名称出现的频率
    print("提取完成，以下为游戏中出现的角色标识，以及每个角色标识对应角色名，还有角色名出现的频率:")
    for key, counter in character_counter.items():
        print(f"{key}: {counter}")


if __name__ == "__main__":
    main(parse_args())
