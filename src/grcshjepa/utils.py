from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tarfile
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Reproducibility is favored for smoke tests. Production runs may relax this after logging.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def config_hash(config: Any) -> str:
    data = asdict(config) if is_dataclass(config) else dict(config)
    payload = json.dumps(data, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]


def save_json(obj: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str))


def git_commit_hash() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "unknown"


def environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": subprocess.check_output(["python", "--version"], text=True).strip(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "git_commit": git_commit_hash(),
    }
    if torch.cuda.is_available():
        report["gpu_name"] = torch.cuda.get_device_name(0)
        report["cuda_version"] = torch.version.cuda
    return report


def write_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    save_json(
        {
            "written_at_utc": datetime.now(timezone.utc).isoformat(),
            "git_commit": git_commit_hash(),
            **payload,
        },
        path,
    )


def archive_directory(source_dir: str | Path, archive_path: str | Path) -> Path:
    source_dir = Path(source_dir)
    archive_path = Path(archive_path)
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_path, "w:gz") as tar:
        tar.add(source_dir, arcname=source_dir.name)
    return archive_path


def copy_to_drive_if_available(source: str | Path, drive_dir: str | Path | None) -> Path | None:
    if drive_dir is None:
        return None
    drive_dir = Path(drive_dir)
    if not drive_dir.exists():
        return None
    source = Path(source)
    target = drive_dir / source.name
    if source.is_dir():
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target
