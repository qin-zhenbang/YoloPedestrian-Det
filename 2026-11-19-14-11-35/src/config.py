"""公共工具：定位项目根目录、加载 exp.yaml、路径解析。

所有脚本只从这里取配置，避免在代码里写死参数。
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp.yaml"


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """加载实验配置。默认读 configs/exp.yaml。"""
    cfg_path = Path(path) if path else CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = PROJECT_ROOT / cfg_path
    with cfg_path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    if not isinstance(config, dict):
        raise ValueError(f"配置文件内容不是字典: {cfg_path}")
    return config


def resolve(path: Path | str) -> Path:
    """把配置里的相对路径解析为相对项目根目录的绝对路径。"""
    p = Path(path)
    return p if p.is_absolute() else PROJECT_ROOT / p
