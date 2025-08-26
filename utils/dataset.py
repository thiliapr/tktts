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
