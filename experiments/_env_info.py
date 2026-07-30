"""
Environment info capture for experiment reproducibility.

Records hardware (GPU, CPU, RAM), software (Python, CUDA, torch),
and runtime parameters alongside experiment results.
"""

import datetime
import os
import platform
import sys


def capture() -> dict:
    """Capture hardware and software environment info."""
    info = {
        "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
        },
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
        },
        "cpu": {
            "count": os.cpu_count(),
        },
        "gpu": _capture_gpu_info(),
        "packages": _capture_packages(),
    }
    return info


def _capture_gpu_info() -> dict:
    """Capture GPU information."""
    result = {"available": False, "devices": []}

    # Try nvidia-smi first (most reliable)
    import subprocess
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap",
             "--format=csv,noheader,nounits"],
            timeout=5, text=True,
        )
        for line in output.strip().split("\n"):
            if not line.strip():
                continue
            parts = [p.strip() for p in line.split(",")]
            result["devices"].append({
                "name": parts[0] if len(parts) > 0 else "unknown",
                "memory_mb": int(parts[1]) if len(parts) > 1 else 0,
                "compute_cap": parts[2] if len(parts) > 2 else "",
            })
        result["available"] = len(result["devices"]) > 0
    except (subprocess.SubprocessError, FileNotFoundError, ValueError):
        pass

    # Try torch as fallback
    if not result["available"]:
        try:
            import torch
            result["available"] = torch.cuda.is_available()
            if result["available"]:
                for i in range(torch.cuda.device_count()):
                    result["devices"].append({
                        "name": torch.cuda.get_device_name(i),
                        "memory_mb": torch.cuda.get_device_properties(i).total_memory // (1024 ** 2),
                    })
        except ImportError:
            pass

    return result


def _capture_packages() -> dict:
    """Capture relevant package versions."""
    packages = {}
    for pkg in ["torch", "transformers", "vllm", "trl", "peft", "accelerate",
                 "sentence-transformers", "chromadb", "numpy", "tiktoken"]:
        try:
            mod = __import__(pkg)
            packages[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            packages[pkg] = None
    return packages


def format_summary(env: dict) -> str:
    """Format environment info as a short summary string."""
    lines = ["Environment:", f"  Python: {env['python']['version'][:30]}"]
    if env['gpu']['available']:
        for d in env['gpu']['devices']:
            lines.append(f"  GPU: {d['name']} ({d.get('memory_mb', 0)}MB)")
    else:
        lines.append("  GPU: None (CI/reference mode)")
    return "\n".join(lines)
