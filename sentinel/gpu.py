"""
gpu.py

Read GPU utilization and memory for the status bar in the demo window.

On a desktop GPU, nvidia-smi returns full data.
On the Jetson Orin NX, nvidia-smi leaves utilization and memory blank, so
read_soc_gpu() reads the SoC load file and /proc/meminfo instead.
read_gpu() tries nvidia-smi first and falls back to the SoC path automatically.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def parse_gpu_line(line: str) -> tuple[int, int, int] | None:
    """Parse 'utilized%, used_MiB, free_MiB' from one nvidia-smi CSV line.

    Returns a tuple of three ints, or None when the line cannot be parsed.
    """
    parts = [part.strip() for part in line.split(",")]
    if len(parts) != 3:
        return None
    try:
        return int(float(parts[0])), int(float(parts[1])), int(float(parts[2]))
    except ValueError:
        return None


def format_gpu_free(free_mib: int) -> str:
    """Express free GPU memory in gigabytes to one decimal place."""
    return f"{free_mib / 1024:.1f} GB"


def read_soc_gpu() -> tuple[int, int, int] | None:
    """Read Jetson Orin NX GPU load from the SoC sysfs file.

    The load file stores 0-1000 (tenths of a percent). Memory is read from
    /proc/meminfo because nvidia-smi reports blank values on the Orin.
    Returns (utilized_percent, used_MiB, free_MiB) or None when unavailable.
    """
    load_file = Path("/sys/devices/platform/gpu.0/load")
    if not load_file.is_file():
        return None
    try:
        # Load file value is 0-1000; divide by 10 to get a 0-100 percent.
        utilized = int(load_file.read_text().strip()) // 10
        meminfo = Path("/proc/meminfo").read_text(encoding="utf-8")
    except (OSError, ValueError):
        return None
    total = available = None
    for line in meminfo.splitlines():
        if line.startswith("MemTotal:"):
            total = int(line.split()[1]) // 1024
        elif line.startswith("MemAvailable:"):
            available = int(line.split()[1]) // 1024
    if total is None or available is None:
        return None
    used = max(0, total - available)
    return utilized, used, available


def read_gpu() -> tuple[int, int, int] | None:
    """Return (utilized_percent, used_MiB, free_MiB) or None when no reading is available.

    Tries nvidia-smi first. Falls back to the Orin SoC sysfs file when
    nvidia-smi is not available or returns blank data.
    """
    try:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=2,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return read_soc_gpu()

    lines = raw.strip().splitlines()
    if not lines:
        return read_soc_gpu()

    parsed = parse_gpu_line(lines[0])
    # On the Orin, nvidia-smi returns blank data. Fall back to the SoC file.
    if parsed is None:
        return read_soc_gpu()
    return parsed
