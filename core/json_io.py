"""
兼容 output_json 解析：部分文件在第一个完整 JSON 后带有时间戳等多余内容，
导致 json.load() 报 Extra data。此处只解析第一个 JSON 对象（所有 output_json 均含 metadata.source_path / orig_t）。
"""
import json
from pathlib import Path
from typing import Any, Union


def loadFirstJson(path: Union[str, Path]) -> Any:
    """
    从文件中只解析第一个 JSON 对象；若文件末尾有多余内容（如时间戳片段）则忽略。
    用于 output_json 等可能带 trailing extra 的文件。
    """
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    dec = json.JSONDecoder()
    obj, _ = dec.raw_decode(raw)
    return obj
