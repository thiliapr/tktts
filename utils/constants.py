# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

# 预处理
AUDIO_SUFFIXES = [".flac", ".wav", ".ogg", ".mp3"]
DEFAULT_FFT_LENGTH = 2048
DEFAULT_SAMPLE_RATE = 24000.
DEFAULT_HOP_LENGTH = int(0.0125 * DEFAULT_SAMPLE_RATE)
DEFAULT_WIN_LENGTH = int(0.05 * DEFAULT_SAMPLE_RATE)

DEFAULT_NUM_HEADS = 8
DEFAULT_DIM_HEAD = 64
DEFAULT_DIM_FEEDFORWARD = DEFAULT_NUM_HEADS * DEFAULT_DIM_HEAD * 4
DEFAULT_NUM_LAYERS = 6

# 训练超参
DEFAULT_LEARNING_RATE = 5e-5  # 学习率
DEFAULT_WEIGHT_DECAY = 1e-2  # 权重衰减（L2正则化）系数
DEFAULT_DROPOUT = 0.1  # Dropout 概率
