# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

from typing import TypedDict


class AudioMetadata(TypedDict):
    "一段音频的元数据"
    text: str
    positive_prompt: list[str]
    negative_prompt: list[str]


class FastAudioMetadata(TypedDict):
    """
    快速训练数据集中，一段音频特征的元数据

    Attributes:
        text: 音频对应的文本编码序列
        positive_prompt: 正向提示词的编码序列
        negative_prompt: 负向提示词的编码序列
        filename: 该音频特征所处文件
        audio_id：该音频特征所处文件标识 ID
    """
    text: list[int]
    positive_prompt: list[int]
    negative_prompt: list[int]
    filename: str
    audio_id: int
