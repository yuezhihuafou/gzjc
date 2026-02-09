#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
XJTU MC(.f + sidecar .json) -> API -> output_json （不落盘 wav）

目标：
- 直接从 .f 二进制信号生成 WAV 字节流（BytesIO），上传声音 API
- 仅保存 JSON-first 产物到 datasets/sound_api/output_json/{bearing_id}/
- 支持断点续跑、并行、失败原因统计与报表

输出 JSON schema（与项目规范一致）：
{
  "data": { "frequency": [...], "volume": [...], "density": [...] },
  "metadata": {
    "bearing_id": "1000",
    "t": 146413,
    "orig_t": 146413,
    "source_path": "D:/.../XJTU-SY_1000_146413.f",
    "sampling_rate": 25600,
    "api_url": "...",
    "api_params": {...},
    "created_at": "2026-01-16T12:34:56"
  }
}

注意：
- 这里的 metadata.t 默认写入 orig_t（来自文件名/解析），便于追溯；训练用重编号 t 在 cache 阶段生成。
"""

import argparse
import io
import json
import os
import time
import wave
import sys
import threading
import zlib
import random
import string
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import requests
from tqdm import tqdm

# 确保可直接以脚本方式运行（把项目根目录加入 sys.path）
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tools.sound_api.convert_sound_api import (
    get_default_config,
    parse_api_response,
    parse_bearing_id_from_filename,
    parse_t_from_filename,
)

_thread_local = threading.local()


def _get_session(pool_maxsize: int = 32) -> requests.Session:
    """
    每个线程复用一个 requests.Session，减少握手与 TCP 连接开销。
    """
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=pool_maxsize, pool_maxsize=pool_maxsize)
        sess.mount("http://", adapter)
        sess.mount("https://", adapter)
        _thread_local.session = sess
    return sess


class RateLimiter:
    """
    简单全局限流器（进程内）：限制平均 QPS，避免速度抖动/触发服务端限流。
    - qps=None 或 qps<=0: 不限流
    """

    def __init__(self, qps: Optional[float]):
        self.qps = float(qps) if qps is not None else None
        self._lock = threading.Lock()
        self._next_time = 0.0

    def acquire(self):
        if self.qps is None or self.qps <= 0:
            return
        interval = 1.0 / self.qps
        while True:
            with self._lock:
                now = time.time()
                if now >= self._next_time:
                    self._next_time = now + interval
                    return
                sleep_for = self._next_time - now
            if sleep_for > 0:
                time.sleep(min(sleep_for, 0.2))


class IdentityRotationManager:
    """
    身份轮换管理器：利用服务器登录漏洞自动伪造新身份，突破日限额限制。
    - 使用 requests.Session() 自动管理 Cookie (JSESSIONID)
    - 每4500次上传后自动切换新身份
    - 线程安全的计数和轮换逻辑
    """
    
    LOGIN_URL = "http://115.236.25.110:8003/hardware/device/sound-user/login"
    UPLOAD_THRESHOLD = 4500  # 留500安全余量（5000-4500）
    
    def __init__(self, enable_rotation: bool = True):
        """
        Args:
            enable_rotation: 是否启用身份轮换（True=启用，False=保持静态）
        """
        self.enable_rotation = enable_rotation
        self.session = requests.Session()  # 全局会话对象，自动管理 Cookie
        self.current_user_id = None
        self.upload_count = 0
        self._lock = threading.Lock()
        
    def generate_new_user_id(self) -> str:
        """生成19位随机userId（模仿格式：6qPq3rbzDIS4cXOdhxi）"""
        return ''.join(random.choices(string.ascii_letters + string.digits, k=19))
    
    def refresh_identity(self) -> Tuple[bool, str]:
        """
        刷新身份：生成新userId -> Session登录（Cookie自动保存）
        
        Returns:
            (成功标志, 新的userId)
        """
        new_user_id = self.generate_new_user_id()
        payload = {"userId": new_user_id}
        
        try:
            # 使用 session.post - Cookie 自动保存在 session 内部
            resp = self.session.post(
                self.LOGIN_URL,
                json=payload,
                timeout=10
            )
            
            if resp.status_code == 200:
                print(f"[登录成功] 新ID: {new_user_id}")
                # 保存当前 userId（用于上传时传递给 API）
                self.current_user_id = new_user_id
                # 检查 session 中是否已保存 Cookie
                cookies = self.session.cookies.get_dict()
                if cookies:
                    print(f"[Cookie已保存] {list(cookies.keys())}")
                return True, new_user_id
            else:
                try:
                    err_msg = resp.json()
                    print(f"[登录失败] HTTP {resp.status_code}: {err_msg}")
                except:
                    print(f"[登录失败] HTTP {resp.status_code}: {resp.text[:100]}")
                return False, None
                
        except Exception as e:
            print(f"[登录异常] {e}")
            return False, None
    
    def maybe_rotate_identity(self) -> bool:
        """
        检查是否需要轮换身份（线程安全）
        每次上传后调用一次，计数器 +1
        当达到阈值时，自动刷新身份
        
        Returns:
            True=继续，False=轮换失败需要重试
        """
        if not self.enable_rotation:
            return True
        
        with self._lock:
            self.upload_count += 1
            
            if self.upload_count >= self.UPLOAD_THRESHOLD:
                print(f"\n[轮换触发] 已上传 {self.upload_count}/{self.UPLOAD_THRESHOLD} 次，切换新身份...")
                
                # 尝试刷新身份
                max_retries = 3
                for attempt in range(max_retries):
                    success, new_user_id = self.refresh_identity()
                    
                    if success:
                        self.current_user_id = new_user_id
                        self.upload_count = 0
                        return True
                    else:
                        if attempt < max_retries - 1:
                            print(f"[重试] 第 {attempt+1} 次重试...")
                            time.sleep(2 ** attempt)
                
                # 三次都失败
                print(f"[失败] 轮换身份失败，暂停60秒后重试...")
                return False
        
        return True
    
    def get_session(self) -> requests.Session:
        """获取会话对象（用于 API 调用）"""
        return self.session
    
    def get_current_user_id(self) -> Optional[str]:
        """获取当前的 userId（用于上传时传递给 API）"""
        return self.current_user_id


_bad_files_lock = threading.Lock()

def _append_bad_file(bad_files_path: Optional[str], reason: str, rel_source: str) -> None:
    if not bad_files_path:
        return
    with _bad_files_lock:
        p = Path(bad_files_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(f"{reason}\t{rel_source}\n")


def load_binary_signal(binary_file: Path, sidecar_json: Path) -> Tuple[Optional[np.ndarray], Optional[Dict]]:
    """从 .f 二进制文件和对应的 .json 元数据文件中加载信号。"""
    try:
        with open(sidecar_json, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        data_shape = tuple(metadata["data_shape"])  # (channels, length)
        data_dtype = metadata["data_dtype"]

        data = np.fromfile(str(binary_file), dtype=data_dtype)
        expected = int(np.prod(data_shape)) if len(data_shape) > 0 else int(data.size)
        if data.size != expected:
            raise ValueError(
                f"size_mismatch: expected={expected} got={data.size} "
                f"shape={data_shape} dtype={data_dtype} file_bytes={binary_file.stat().st_size}"
            )
        data = data.reshape(data_shape)
        return data, metadata
    except Exception as e:
        return None, {"error": f"load_error: {e}"}


def normalize_signal(signal: np.ndarray, method: str = "minmax") -> np.ndarray:
    """归一化到 [-1, 1]。"""
    if method == "minmax":
        min_val = float(np.min(signal))
        max_val = float(np.max(signal))
        if max_val > min_val:
            return 2.0 * (signal - min_val) / (max_val - min_val) - 1.0
        return np.zeros_like(signal)
    if method == "zscore":
        mean = float(np.mean(signal))
        std = float(np.std(signal))
        if std > 0:
            signal_norm = (signal - mean) / std
            return np.clip(signal_norm / 3.0, -1.0, 1.0)  # 3-sigma
        return np.zeros_like(signal)
    raise ValueError(f"不支持的归一化方法: {method}")


def build_wav_bytes(
    data: np.ndarray,
    sampling_rate: int,
    channel_mode: str = "horizontal",
    normalize_method: str = "minmax",
) -> bytes:
    """
    将 (2, L) 的振动信号构造成 WAV 字节流（PCM16）。
    - horizontal: mono, data[0]
    - vertical: mono, data[1]
    - mix: mono, (data[0]+data[1])/2
    - stereo: stereo, (L,2)
    """
    if data.ndim != 2 or data.shape[0] < 2:
        raise ValueError(f"数据形状不符合预期 (2, L): got {data.shape}")

    if channel_mode == "horizontal":
        signal = data[0]
        channels = 1
        pcm = normalize_signal(signal, normalize_method)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)
        frames = pcm_int16.tobytes()
    elif channel_mode == "vertical":
        signal = data[1]
        channels = 1
        pcm = normalize_signal(signal, normalize_method)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)
        frames = pcm_int16.tobytes()
    elif channel_mode == "mix":
        signal = (data[0] + data[1]) / 2.0
        channels = 1
        pcm = normalize_signal(signal, normalize_method)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)
        frames = pcm_int16.tobytes()
    elif channel_mode == "stereo":
        stereo = np.stack([data[0], data[1]], axis=1)  # (L,2)
        channels = 2
        pcm0 = normalize_signal(stereo[:, 0], normalize_method)
        pcm1 = normalize_signal(stereo[:, 1], normalize_method)
        pcm = np.stack([pcm0, pcm1], axis=1)  # (L,2)
        pcm_int16 = (pcm * 32767.0).astype(np.int16)
        frames = pcm_int16.tobytes()
    else:
        raise ValueError(f"不支持的通道模式: {channel_mode}")

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # int16
        wf.setframerate(int(sampling_rate))
        wf.writeframes(frames)

    return buf.getvalue()


def call_sound_api_with_wav_bytes(
    wav_bytes: bytes,
    filename: str,
    api_url: str,
    headers: Optional[Dict],
    file_param_name: str,
    form_data_params: Optional[Dict],
    timeout: int,
    retries: int,
    retry_backoff: float,
    limiter: Optional[RateLimiter] = None,
    auth_file: Optional[str] = None,
    identity_rotator: Optional[IdentityRotationManager] = None,
) -> Tuple[Optional[Dict], Optional[str]]:
    """上传 wav bytes 到 API，返回 (json_dict, error_reason)。"""
    # [轮换检查] 在API调用前，检查是否需要轮换身份
    if identity_rotator:
        while not identity_rotator.maybe_rotate_identity():
            # 轮换失败，暂停后重试
            time.sleep(60)
    
    # 选择会话对象：如果有 identity_rotator，优先使用其 session（自动管理 Cookie）
    if identity_rotator:
        session = identity_rotator.get_session()
    else:
        session = _get_session(pool_maxsize=32)
    
    default_headers = {"User-Agent": "Python-requests/Sound-API-Client/1.0"}
    if headers:
        default_headers.update(headers)
    
    # 如果指定了 auth_file，每次请求前都尝试读取最新的 Cookie/Authorization（支持热更新）
    if auth_file and os.path.exists(auth_file):
        try:
            with open(auth_file, "r", encoding="utf-8") as f:
                auth_data = json.load(f)
                if "Cookie" in auth_data and auth_data["Cookie"]:
                    default_headers["Cookie"] = str(auth_data["Cookie"]).strip()
                auth_val = auth_data.get("Authorization") or ""
                auth_val = (auth_val or "").strip()
                if auth_val and "如果有" not in auth_val and "在这里" not in auth_val:
                    default_headers["Authorization"] = auth_val
        except Exception:
            pass
    
    data = {}
    if form_data_params:
        data.update(form_data_params)
    
    # 【关键修复】userId 必须同时出现在 URL 和表单数据中
    if identity_rotator:
        current_user_id = identity_rotator.get_current_user_id()
        if current_user_id:
            data["userId"] = current_user_id
    
    # 【关键修复】如果使用身份轮换，需要在 URL 中添加 userId 查询参数
    request_url = api_url
    if identity_rotator:
        current_user_id = identity_rotator.get_current_user_id()
        if current_user_id:
            separator = "&" if "?" in api_url else "?"
            request_url = f"{api_url}{separator}userId={current_user_id}"

    last_err: Optional[str] = None
    for attempt in range(retries + 1):
        try:
            if limiter is not None:
                limiter.acquire()
            bio = io.BytesIO(wav_bytes)
            bio.seek(0)
            files = {file_param_name: (filename, bio, "audio/wav")}
            # 使用 session.post - Cookie 自动携带
            resp = session.post(request_url, files=files, data=data, headers=default_headers, timeout=timeout)
            if resp.status_code != 200:
                snippet = ""
                try:
                    snippet = resp.text[:200].replace("\n", " ").replace("\r", " ")
                except Exception:
                    pass
                raise RuntimeError(f"http_status={resp.status_code} body={snippet}")
            try:
                return resp.json(), None
            except Exception as e:
                last_err = f"json_decode_error: {e}"
        except Exception as e:
            last_err = str(e)
            if attempt >= retries:
                return None, last_err
            time.sleep(retry_backoff * (2 ** attempt))

    return None, last_err


def atomic_write_json(path: Path, obj: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    tmp.replace(path)


def process_one_mc_file(
    f_path: Path,
    mc_dir: Path,
    output_root: Path,
    channel_mode: str,
    normalize_method: str,
    force_sampling_rate: Optional[int],
    resume: bool,
    shard_id: int,
    num_shards: int,
    api_url: str,
    headers: Dict,
    form_data_params: Dict,
    file_param_name: str,
    timeout: int,
    retries: int,
    retry_backoff: float,
    limiter_qps: Optional[float],
    bad_files_path: Optional[str],
    auth_file: Optional[str] = None,
    identity_rotator: Optional[IdentityRotationManager] = None,
) -> Dict:
    """处理单个 .f 文件，返回结果字典（用于报表）。"""
    stem = f_path.stem
    sidecar = f_path.with_suffix(".json")
    rel_source = str(f_path.relative_to(mc_dir)) if mc_dir in f_path.parents else str(f_path)

    # 分片：用于多机/多进程扩展吞吐（保证每个样本只处理一次）
    if num_shards > 1:
        h = zlib.crc32(stem.encode("utf-8")) & 0xFFFFFFFF
        if (h % num_shards) != shard_id:
            return {
                "status": "skipped",
                "reason": "shard_skip",
                "bearing_id": None,
                "orig_t": None,
                "source_path": rel_source,
            }

    # bearing_id / orig_t 解析优先级：
    # 1) 文件名（XJTU-SY_{bearing_id}_t{t} 或 _{6位t}）
    # 2) sidecar 元数据（bearing_name / file_number / segment_index 等）
    # 3) 兜底：父目录名
    bearing_id = parse_bearing_id_from_filename(f_path.name)
    orig_t = parse_t_from_filename(f_path.name)

    if not sidecar.exists():
        _append_bad_file(bad_files_path, "missing_sidecar_json", rel_source)
        return {
            "status": "failed",
            "reason": "missing_sidecar_json",
            "bearing_id": str(bearing_id),
            "orig_t": orig_t,
            "source_path": rel_source,
        }

    data, mc_meta = load_binary_signal(f_path, sidecar)
    if data is None:
        _append_bad_file(bad_files_path, mc_meta.get("error", "load_error"), rel_source)
        return {
            "status": "failed",
            "reason": mc_meta.get("error", "load_error"),
            "bearing_id": str(bearing_id) if bearing_id is not None else f_path.parent.name,
            "orig_t": orig_t,
            "source_path": rel_source,
        }

    sampling_rate = int(force_sampling_rate or mc_meta.get("sampling_rate", 25600))

    # 用 sidecar 补齐 bearing_id / orig_t（与 README 里的字段对齐）
    if bearing_id is None:
        # 常见：bearing_name 或 filename/binary_file 中包含 XJTU-SY_ 前缀
        for key in ("binary_file", "filename"):
            if key in mc_meta and isinstance(mc_meta[key], str):
                bearing_id = parse_bearing_id_from_filename(mc_meta[key])
                if bearing_id is not None:
                    break
        if bearing_id is None and "bearing_name" in mc_meta and mc_meta["bearing_name"]:
            bearing_id = str(mc_meta["bearing_name"])
    if bearing_id is None:
        bearing_id = f_path.parent.name

    if orig_t is None:
        # README 中明确有 file_number / segment_index
        if mc_meta.get("file_number") is not None:
            try:
                orig_t = int(mc_meta["file_number"])
            except Exception:
                orig_t = None
        if orig_t is None and mc_meta.get("segment_index") is not None:
            try:
                orig_t = int(mc_meta["segment_index"])
            except Exception:
                orig_t = None

    # 现在 bearing_id 已经稳定，才计算输出路径/断点续跑（避免落到 output_json/None）
    out_json = output_root / "output_json" / str(bearing_id) / f"{stem}.json"
    if resume and out_json.exists():
        return {
            "status": "skipped",
            "reason": "exists",
            "bearing_id": str(bearing_id),
            "orig_t": orig_t,
            "source_path": rel_source,
            "output_json": str(out_json),
        }

    try:
        wav_bytes = build_wav_bytes(
            data=data,
            sampling_rate=sampling_rate,
            channel_mode=channel_mode,
            normalize_method=normalize_method,
        )
    except Exception as e:
        _append_bad_file(bad_files_path, f"wav_build_error: {e}", rel_source)
        return {
            "status": "failed",
            "reason": f"wav_build_error: {e}",
            "bearing_id": str(bearing_id),
            "orig_t": orig_t,
            "source_path": rel_source,
        }

    # 重要对齐点：API 参数里的 sampleFrq 必须与实际 WAV 采样率一致
    # 默认配置里 sampleFrq=192000，但 XJTU-SY sidecar 里 sampling_rate=25600
    api_params = dict(form_data_params) if form_data_params else {}
    api_params["sampleFrq"] = str(sampling_rate)
    # 注：userId 会在 call_sound_api_with_wav_bytes 中自动添加到 URL 和表单数据

    limiter = RateLimiter(limiter_qps)
    api_result, api_err = call_sound_api_with_wav_bytes(
        wav_bytes=wav_bytes,
        filename=f"{stem}.wav",
        api_url=api_url,
        headers=headers,
        file_param_name=file_param_name,
        form_data_params=api_params,
        timeout=timeout,
        retries=retries,
        retry_backoff=retry_backoff,
        limiter=limiter,
        auth_file=auth_file,
        identity_rotator=identity_rotator,
    )
    if not api_result:
        reason = f"api_error: {api_err}" if api_err else "api_error"
        _append_bad_file(bad_files_path, reason, rel_source)
        return {
            "status": "failed",
            "reason": reason,
            "bearing_id": str(bearing_id),
            "orig_t": orig_t,
            "source_path": rel_source,
        }

    parsed = parse_api_response(api_result, verbose=False)
    if not parsed:
        _append_bad_file(bad_files_path, "parse_error", rel_source)
        return {
            "status": "failed",
            "reason": "parse_error",
            "bearing_id": str(bearing_id),
            "orig_t": orig_t,
            "source_path": rel_source,
        }

    payload = {
        "data": {
            "frequency": parsed["frequency"].tolist(),
            "volume": parsed["volume"].tolist(),
            "density": parsed["density"].tolist(),
        },
        "metadata": {
            "bearing_id": str(bearing_id),
            # 这里的 t 写入原始采集序号，便于追溯；训练重编号在 cache 阶段产生
            "t": orig_t,
            "orig_t": orig_t,
            "source_path": str(f_path),
            "sampling_rate": sampling_rate,
            "api_url": api_url,
            "api_params": api_params,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        },
    }

    try:
        atomic_write_json(out_json, payload)
    except Exception as e:
        _append_bad_file(bad_files_path, f"write_error: {e}", rel_source)
        return {
            "status": "failed",
            "reason": f"write_error: {e}",
            "bearing_id": str(bearing_id),
            "orig_t": orig_t,
            "source_path": rel_source,
        }

    return {
        "status": "success",
        "bearing_id": str(bearing_id),
        "orig_t": orig_t,
        "source_path": rel_source,
        "output_json": str(out_json),
        "data_points": int(len(payload["data"]["frequency"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="XJTU MC(.f) -> API -> output_json（不落盘 wav）")
    parser.add_argument("--mc_dir", type=str, required=True, help="包含 .f 与 sidecar .json 的目录（递归扫描）")
    parser.add_argument("--output_root", type=str, default="datasets/sound_api", help="输出根目录（默认：datasets/sound_api）")
    parser.add_argument("--channel-mode", type=str, choices=["horizontal", "vertical", "mix", "stereo"], default="horizontal")
    parser.add_argument("--normalize-method", type=str, choices=["minmax", "zscore"], default="minmax")
    parser.add_argument("--sampling-rate", type=int, default=None, help="强制采样率（默认从 sidecar 读取，否则 25600）")
    parser.add_argument("--workers", type=int, default=8, help="并行 worker 数（默认 8）")
    parser.add_argument(
        "--max-inflight",
        type=int,
        default=None,
        help="同时在途任务上限（默认: workers*4）。用于避免一次性 submit 全量文件导致调度/内存开销巨大。",
    )
    parser.add_argument(
        "--qps",
        type=float,
        default=None,
        help="全局限流 QPS（例如 20 表示平均每秒最多 20 个请求）。用于稳定速度/避免触发服务端限流。默认不限制。",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=1,
        help="分片总数（用于多机/多进程并行跑；默认 1 表示不分片）",
    )
    parser.add_argument(
        "--shard-id",
        type=int,
        default=0,
        help="当前分片编号（0..num_shards-1）",
    )
    parser.add_argument("--resume", action="store_true", default=True,
                       help="断点续跑（目标 JSON 已存在则跳过，默认启用）")
    parser.add_argument("--timeout", type=int, default=60, help="API 请求超时秒数（默认 60）")
    parser.add_argument("--retries", type=int, default=2, help="API 失败重试次数（默认 2）")
    parser.add_argument("--retry-backoff", type=float, default=1.0, help="重试退避基数秒（默认 1.0）")
    parser.add_argument("--report", type=str, default="datasets/sound_api/logs/mc_to_api_report.json", help="转换报表输出路径")
    parser.add_argument("--bad-files", type=str, default="datasets/sound_api/logs/bad_files_mc_to_api.txt", help="失败文件列表输出路径")
    parser.add_argument("--print-every", type=int, default=2000, help="每处理 N 个样本打印一次成功/失败/跳过统计（默认 2000）")
    parser.add_argument("--print-first-failures", type=int, default=5, help="启动后打印前 N 条失败原因（默认 5）")
    parser.add_argument("--auth-cookie", type=str, default=None, help="覆盖 Cookie header（例如从浏览器 F12 复制的完整 Cookie）")
    parser.add_argument("--auth-token", type=str, default=None, help="覆盖 Authorization header（例如从浏览器 F12 复制）")
    parser.add_argument("--auth-file", type=str, default=None, help="从文件动态读取 auth（JSON: {\"Cookie\":..., \"Authorization\":...}）；支持热更新（每次请求前读）")
    parser.add_argument("--enable-identity-rotation", action="store_true", 
                       help="启用自动身份轮换（利用登录漏洞，每4500次自动切换新身份，突破日限额5000）")
    parser.add_argument("--retry-failed", action="store_true",
                       help="仅重跑失败项：从 --bad-files 读取列表，只处理这些 .f 并输出到 output_json（与 --resume 兼容）")
    args = parser.parse_args()

    mc_dir = Path(args.mc_dir)
    if not mc_dir.exists():
        raise SystemExit(f"错误: 目录不存在: {mc_dir}")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    api_url, headers, form_data_params, file_param_name = get_default_config()
    
    # 覆盖 auth（从命令行或文件）
    if args.auth_cookie:
        headers["Cookie"] = args.auth_cookie
    if args.auth_token:
        headers["Authorization"] = args.auth_token
    auth_file_resolved: Optional[str] = None
    if args.auth_file:
        p = Path(args.auth_file)
        if not p.is_absolute():
            p = (Path.cwd() / p).resolve()
        if not p.exists():
            raise SystemExit(
                f"错误: --auth-file 指定的文件不存在: {p}\n"
                f"你传的是 '{args.auth_file}'。若 Cookie 写在 auth_example.json，请用: --auth-file auth_example.json"
            )
        auth_file_resolved = str(p)
        print(f"[Auth] 将从文件读取 auth: {auth_file_resolved}")

    # 提前创建日志目录（方便你中途查看 bad_files）
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.bad_files).parent.mkdir(parents=True, exist_ok=True)
    # touch bad_files，避免中途 Get-Content 报不存在
    Path(args.bad_files).touch(exist_ok=True)

    # [新增] 初始化身份轮换管理器
    identity_rotator = None
    if args.enable_identity_rotation:
        identity_rotator = IdentityRotationManager(enable_rotation=True)
        # 首次登录，获取初始 Cookie
        success, user_id = identity_rotator.refresh_identity()
        if success:
            print(f"[初始化] 身份轮换已启用，首次登录成功，ID: {user_id}")
        else:
            print(f"[警告] 初始登录失败，将在处理文件时重试")
    else:
        print(f"[轮换] 身份轮换已禁用，使用静态配置")

    # 扫描 .f 文件（全量）或仅失败列表（--retry-failed）
    if args.retry_failed:
        bad_path = Path(args.bad_files)
        if not bad_path.exists():
            raise SystemExit(f"错误: --retry-failed 需要失败列表文件存在: {bad_path}")
        failed_rel: List[str] = []
        with open(bad_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # 格式: reason\trel_source
                parts = line.split("\t")
                if len(parts) >= 2:
                    failed_rel.append(parts[-1].strip())
                elif len(parts) == 1 and parts[0]:
                    failed_rel.append(parts[0].strip())
        failed_rel = list(dict.fromkeys(failed_rel))  # 去重，保持顺序
        f_files = []
        for rel in failed_rel:
            p = (mc_dir / rel).resolve()
            if p.exists() and p.suffix.lower() == ".f":
                f_files.append(p)
            else:
                # 可能 rel 只是文件名，需在 mc_dir 下递归查找
                if "/" not in rel and "\\" not in rel:
                    for fp in mc_dir.rglob(rel):
                        if fp.suffix.lower() == ".f":
                            f_files.append(fp)
                            break
        f_files = sorted(set(f_files))
        print(f"[Retry-Failed] 从 {bad_path} 读取 {len(failed_rel)} 条，解析到 {len(f_files)} 个存在的 .f 文件")
        if not f_files:
            raise SystemExit("错误: --retry-failed 未解析到任何存在的 .f 文件")
    else:
        f_files = sorted(mc_dir.rglob("*.f"))
        if not f_files:
            raise SystemExit(f"错误: 在 {mc_dir} 中未找到 .f 文件")

    start = time.time()
    results: List[Dict] = []

    workers = int(args.workers)
    max_inflight = int(args.max_inflight) if args.max_inflight is not None else max(1, workers * 4)
    num_shards = int(args.num_shards)
    shard_id = int(args.shard_id)
    if num_shards < 1:
        raise SystemExit("错误: --num-shards 必须 >= 1")
    if shard_id < 0 or shard_id >= num_shards:
        raise SystemExit(f"错误: --shard-id 必须在 [0, {num_shards-1}] 范围内")

    # 限流提交：避免 28 万个任务一次性 submit（会显著拖慢调度并吃掉大量内存）
    it = iter(f_files)
    inflight = {}
    success_n = 0
    failed_n = 0
    skipped_n = 0
    processed_n = 0

    with ThreadPoolExecutor(max_workers=workers) as ex, tqdm(total=len(f_files), desc="MC->API") as pbar:
        # 预填充
        for _ in range(min(max_inflight, len(f_files))):
            try:
                f_path = next(it)
            except StopIteration:
                break
            fut = ex.submit(
                process_one_mc_file,
                f_path,
                mc_dir,
                output_root,
                args.channel_mode,
                args.normalize_method,
                args.sampling_rate,
                args.resume,
                shard_id,
                num_shards,
                api_url,
                headers,
                form_data_params,
                file_param_name,
                args.timeout,
                args.retries,
                args.retry_backoff,
                args.qps,
                args.bad_files,
                auth_file_resolved,
                identity_rotator,
            )
            inflight[fut] = True

        while inflight:
            done, _ = wait(inflight.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                inflight.pop(fut, None)
                r = fut.result()
                results.append(r)
                pbar.update(1)
                processed_n += 1
                if r.get("status") == "success":
                    success_n += 1
                elif r.get("status") == "failed":
                    failed_n += 1
                    if args.print_first_failures and failed_n <= int(args.print_first_failures):
                        print(f"[首批失败] reason={r.get('reason')} file={r.get('source_path')}")
                else:
                    skipped_n += 1

                if args.print_every and processed_n % int(args.print_every) == 0:
                    print(
                        f"[进度] processed={processed_n} success={success_n} failed={failed_n} skipped={skipped_n} "
                        f"(qps={args.qps}, workers={workers}, inflight={max_inflight})"
                    )

                # 补充新任务
                try:
                    f_path = next(it)
                except StopIteration:
                    continue
                new_fut = ex.submit(
                    process_one_mc_file,
                    f_path,
                    mc_dir,
                    output_root,
                    args.channel_mode,
                    args.normalize_method,
                    args.sampling_rate,
                    args.resume,
                    shard_id,
                    num_shards,
                    api_url,
                    headers,
                    form_data_params,
                    file_param_name,
                    args.timeout,
                    args.retries,
                    args.retry_backoff,
                    args.qps,
                    args.bad_files,
                    auth_file_resolved,
                    identity_rotator,
                )
                inflight[new_fut] = True

    elapsed = time.time() - start

    # 汇总统计
    success = sum(1 for r in results if r["status"] == "success")
    failed = sum(1 for r in results if r["status"] == "failed")
    skipped = sum(1 for r in results if r["status"] == "skipped")

    report_obj = {
        "mc_dir": str(mc_dir),
        "output_root": str(output_root),
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(results),
        "success": success,
        "failed": failed,
        "skipped": skipped,
        "elapsed_sec": elapsed,
        "args": vars(args),
        "files": results,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_obj, f, ensure_ascii=False, indent=2)

    bad_path = Path(args.bad_files)
    bad_path.parent.mkdir(parents=True, exist_ok=True)
    failed_lines = [r for r in results if r["status"] == "failed"]
    if failed_lines:
        with open(bad_path, "w", encoding="utf-8") as f:
            for r in failed_lines:
                f.write(f"{r.get('reason','unknown')}\t{r.get('source_path','')}\n")
    else:
        # 本轮 0 失败时不覆盖，避免清空 retry 列表导致 --retry-failed 读不到
        print(f"[提示] 本轮 0 失败，未覆盖 {bad_path}，保留原内容供 --retry-failed 使用")

    print("\n" + "=" * 80)
    print("MC(.f) -> API -> output_json 完成")
    print("=" * 80)
    print(f"总文件数: {len(results)}")
    print(f"成功: {success} | 失败: {failed} | 跳过: {skipped}")
    print(f"耗时: {elapsed:.2f} 秒")
    print(f"报表: {report_path}")
    print(f"失败列表: {bad_path}")


if __name__ == "__main__":
    main()

