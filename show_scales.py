# 本文件是 TkTTS 的一部分
# SPDX-FileCopyrightText: 2025 thiliapr <thiliapr@tutanota.com>
# SPDX-FileContributor: thiliapr <thiliapr@tutanota.com>
# SPDX-License-Identifier: AGPL-3.0-or-later

import pathlib
import argparse
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
    parser = argparse.ArgumentParser(description="按数据流向展示模型每一层的缩放因子")
    parser.add_argument("ckpt_path", type=pathlib.Path, help="检查点路径")
    return parser.parse_args(args)


def main(args: argparse.Namespace):
    _, model_state, _, model_config = load_checkpoint(args.ckpt_path)

    for layer_idx in range(model_config.num_layers):
        print(f"encoder[{layer_idx:03d}].attn_scale       {model_state[f'encoder.{layer_idx}.attention_scale'].item():.3f}")
        print(f"encoder[{layer_idx:03d}].ff_scale         {model_state[f'encoder.{layer_idx}.feedforward_scale'].item():.3f}")

    for layer_idx in range(model_config.num_layers):
        print(f"decoder[{layer_idx:03d}].self_attn_scale  {model_state[f'decoder.{layer_idx}.self_attention_scale'].item():.3f}")
        print(f"decoder[{layer_idx:03d}].cross_attn_scale {model_state[f'decoder.{layer_idx}.cross_attention_scale'].item():.3f}")
        print(f"decoder[{layer_idx:03d}].ff_scale         {model_state[f'decoder.{layer_idx}.feedforward_scale'].item():.3f}")


if __name__ == "__main__":
    main(parse_args())
