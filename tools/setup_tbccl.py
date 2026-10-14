#!/usr/bin/env python3
# pyright: basic
"""Check this checkout for the supported TBCCL installation, optionally run the locked sync, and print first-run diagnostics. Standard library only.

    setup_tbccl.py check     (default) read-only: platform, required tools, which checkout this is, dashboard, adapter; prints the exact commands to run next
    setup_tbccl.py sync      the same checks, then `uv sync --no-dev --locked --extra <mlx extra> --extra tbccl` in this checkout, then the adapter check
    setup_tbccl.py doctor    read-only, inside the installed environment: exo and adapter versions, the TBCCL runtime, device backend, model directories, namespace and ports

It never clones another revision, never replaces this checkout, never installs system packages (no Node, Rust or Xcode), never touches network interfaces or
firewalls, and `check` and `doctor` change nothing. The supported configuration is the immutable tag `tbccl-integration-v2` with the optional `tbccl` extra.
"""

import argparse
import os
import platform
import shutil
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TAG_PREFIX = "tbccl-integration-v"
SUPPORTED_TAG = "tbccl-integration-v2"
PORTS = (
    ("UDP", 52413, "discovery"),
    ("TCP", 52414, "cluster transport"),
    ("TCP", 52415, "API"),
)

DOCTOR_PROBE = r"""
import json, os, sys
out = {}
try:
    import exo
    out["exo"] = exo.__version__
except Exception as e:
    out["exo"] = f"unavailable ({type(e).__name__}: {e})"
try:
    import exo_tbccl
    ok, why = exo_tbccl.is_available()
    out["adapter"] = {"version": exo_tbccl.__version__, "available": ok, "reason": why}
    from exo_tbccl._loader import native
    out["runtime"] = {n: getattr(native, n)() for n in dir(native) if ("version" in n.lower() or n == "abi_version") and callable(getattr(native, n))}
except Exception as e:
    out["adapter"] = f"not installed or not loadable ({type(e).__name__}: {e})"
try:
    import mlx.core as mx
    out["mlx"] = {"version": getattr(mx, "__version__", "?"), "default_device": str(mx.default_device())}
except Exception as e:
    out["mlx"] = f"unavailable ({type(e).__name__}: {e})"
try:
    from exo.shared import constants as c
    out["models_dirs"] = [str(p) for p in c.EXO_MODELS_DIRS]
    out["models_read_only_dirs"] = [str(p) for p in c.EXO_MODELS_READ_ONLY_DIRS]
    out["offline"] = bool(c.EXO_OFFLINE)
except Exception as e:
    out["models_dirs"] = f"unavailable ({type(e).__name__}: {e})"
print(json.dumps(out))
"""


def run(cmd, **kw):
    try:
        return subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=kw.pop("timeout", 60),
            **kw,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def first_line(r):
    return ((r.stdout or r.stderr).strip().splitlines() or [""])[0] if r else ""


def stdout_line(r):
    """First line of stdout of a successful command, else '' (git describe fails on untagged revisions)."""
    return (r.stdout.strip().splitlines() or [""])[0] if r and r.returncode == 0 else ""


def mlx_extra():
    system, machine = platform.system(), platform.machine()
    if system == "Linux" and machine == "x86_64":
        return (
            "mlx-cuda13",
            "Linux x86-64 (CUDA 13 build of MLX; the adapter wheel exists for this platform)",
        )
    if system == "Darwin" and machine == "arm64":
        return "mlx", "macOS arm64 (Metal)"
    return None, f"{system} {machine}"


def port_state(proto, port):
    kind = socket.SOCK_DGRAM if proto == "UDP" else socket.SOCK_STREAM
    s = socket.socket(socket.AF_INET, kind)
    try:
        s.bind(("0.0.0.0", port))
        return "free"
    except OSError as e:
        return f"in use or not bindable ({e.strerror})"
    finally:
        s.close()


class Report:
    def __init__(self):
        self.missing = []

    def ok(self, label, detail=""):
        print(f"  ok       {label}{': ' + detail if detail else ''}")

    def info(self, label, detail=""):
        print(f"  note     {label}{': ' + detail if detail else ''}")

    def need(self, label, how):
        self.missing.append(label)
        print(f"  MISSING  {label}: {how}")


def check(report):
    extra, plat = mlx_extra()
    print(f"platform: {plat}")
    if extra is None:
        print(
            "  This platform has no exo-tbccl wheel (only Linux x86-64 and macOS arm64). The `tbccl` extra installs nothing here and `MlxTbccl` placement would fail\n  with 'MlxTbccl requires the optional exo-tbccl package'. Plain exo still works; see README for upstream installation."
        )
        return None
    print("required tools")
    uv = shutil.which("uv")
    report.ok("uv", first_line(run([uv, "--version"]))) if uv else report.need(
        "uv",
        "install uv (https://docs.astral.sh/uv/; it also provides CPython 3.13 on demand) or add its directory to PATH (non-interactive ssh shells often lack it)",
    )
    report.ok("git", first_line(run(["git", "--version"]))) if shutil.which(
        "git"
    ) else report.need(
        "git",
        "needed to build MLX from its pinned git revision on macOS and to identify this checkout",
    )
    if shutil.which("cargo") and shutil.which("rustc"):
        report.ok("Rust toolchain", first_line(run(["rustc", "--version"])))
    else:
        report.need(
            "Rust toolchain (cargo, rustc)",
            "exo's Rust bindings are built by `uv sync`; install via https://rustup.rs/",
        )
    if platform.system() == "Darwin":
        if run(["xcode-select", "-p"]) and run(["xcode-select", "-p"]).returncode == 0:
            report.ok(
                "Xcode command line tools", first_line(run(["xcode-select", "-p"]))
            )
        else:
            report.need(
                "Xcode command line tools",
                "run `xcode-select --install`; the first sync compiles MLX from git",
            )
        metal = run(["xcrun", "-f", "metal"])
        report.ok(
            "Metal compiler", first_line(metal)
        ) if metal and metal.returncode == 0 else report.need(
            "Metal toolchain",
            "install Xcode, then `xcodebuild -downloadComponent MetalToolchain` if `xcrun -f metal` fails",
        )
        report.info(
            "first sync", "builds MLX from its pinned git revision: several minutes"
        )
    else:
        smi = shutil.which("nvidia-smi")
        report.ok(
            "NVIDIA driver",
            first_line(
                run([smi, "--query-gpu=name,driver_version", "--format=csv,noheader"])
            ),
        ) if smi else report.info(
            "NVIDIA driver",
            "nvidia-smi not found; the CUDA 13 MLX build needs an NVIDIA GPU and driver at run time",
        )
    print("this checkout")
    tag = stdout_line(
        run(["git", "describe", "--tags", "--exact-match", "HEAD"], cwd=ROOT)
    )
    commit = stdout_line(run(["git", "rev-parse", "HEAD"], cwd=ROOT))
    dirty = bool(stdout_line(run(["git", "status", "--porcelain"], cwd=ROOT)))
    if tag == SUPPORTED_TAG:
        report.ok("revision", f"{tag} ({commit[:12]})")
    elif tag.startswith(TAG_PREFIX):
        report.info(
            "revision",
            f"{tag} ({commit[:12]}); the current supported tag is {SUPPORTED_TAG}. v1 has no `tbccl` extra: use its manual-wheel instructions",
        )
    else:
        report.info(
            "revision",
            f"{tag or 'not a release tag'} ({commit[:12] or 'no git metadata'}); the supported configuration is the tag {SUPPORTED_TAG}, not a branch",
        )
    if dirty:
        report.info("working tree", "has uncommitted changes")
    dash = ROOT / "dashboard" / "build" / "index.html"
    if dash.exists():
        report.ok("dashboard build", str(dash.parent))
    else:
        report.info(
            "dashboard",
            "not built. `exo --help`/`--version` work, but serving needs a dashboard directory. Either `cd dashboard && npm install && npm run build` (needs Node)\n             or, for the HTTP API only, point EXO_DASHBOARD_DIR at an existing empty directory (relative to $HOME or absolute)",
        )
    venv_py = ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        r = run(
            [
                str(venv_py),
                "-c",
                "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())",
            ]
        )
        report.ok("adapter", first_line(r)) if r and r.returncode == 0 else report.info(
            "adapter",
            "environment exists but exo_tbccl is not importable: run the sync below",
        )
    else:
        report.info("environment", "no .venv yet")
    cmd = f"uv sync --no-dev --locked --extra {extra} --extra tbccl"
    print(
        f"\nnext (in {ROOT}):\n  {cmd}\n  uv run --no-sync python -c \"import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())\"    # 0.3.0 (True, '')\n  EXO_ZENOH_NAMESPACE=<same on every node> uv run --no-sync exo\nOr let this script run the sync: python3 tools/setup_tbccl.py sync"
    )
    return extra


def cmd_check(_):
    report = Report()
    extra = check(report)
    if extra is None:
        return 2
    if report.missing:
        print(f"\n{len(report.missing)} prerequisite(s) missing; nothing was changed.")
        return 1
    print("\nprerequisites look fine; nothing was changed.")
    return 0


def cmd_sync(args):
    report = Report()
    extra = check(report)
    if extra is None:
        return 2
    if report.missing:
        print("\nresolve the missing prerequisites first; nothing was changed.")
        return 1
    cmd = ["uv", "sync", "--no-dev", "--locked", "--extra", extra, "--extra", "tbccl"]
    print(f"\nrunning: {' '.join(cmd)}")
    if subprocess.run(cmd, cwd=ROOT, check=False).returncode != 0:
        print(
            "uv sync failed (see its output above). The environment may be partly built; rerun after fixing the cause.",
            file=sys.stderr,
        )
        return 5
    r = run(
        [
            str(ROOT / ".venv" / "bin" / "python"),
            "-c",
            "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())",
        ]
    )
    print(f"adapter: {first_line(r)}")
    return 0 if r and r.returncode == 0 and "True" in r.stdout else 5


def cmd_doctor(_):
    import json

    venv_py = ROOT / ".venv" / "bin" / "python"
    print(f"checkout: {ROOT}")
    if not venv_py.exists():
        print(
            "no .venv here: run `python3 tools/setup_tbccl.py check`, then the sync it prints."
        )
        return 1
    r = run([str(venv_py), "-c", DOCTOR_PROBE], timeout=300)
    if not r or r.returncode != 0:
        print(f"could not inspect the environment: {first_line(r)}")
        return 1
    info = json.loads(r.stdout.strip().splitlines()[-1])
    for key in (
        "exo",
        "adapter",
        "runtime",
        "mlx",
        "models_dirs",
        "models_read_only_dirs",
        "offline",
    ):
        print(f"  {key}: {info.get(key)}")
    ns = os.environ.get("EXO_ZENOH_NAMESPACE") or os.environ.get("EXO_LIBP2P_NAMESPACE")
    print(
        f"  discovery namespace: {ns or '(default: the exo version; set EXO_ZENOH_NAMESPACE identically on every node)'}"
    )
    print(
        f"  EXO_HOME: {os.environ.get('EXO_HOME', '(default)')}   EXO_DASHBOARD_DIR: {os.environ.get('EXO_DASHBOARD_DIR', '(default: dashboard/build)')}"
    )
    for proto, port, what in PORTS:
        print(f"  {proto} {port} ({what}): {port_state(proto, port)}")
    print(
        "common problems: `exo_tbccl is not installed` -> run the sync with --extra tbccl on every node; discovery fails from a detached tmux/nohup session on macOS -> start exo from a live terminal or `ssh -tt`;\n  nodes do not see each other -> same EXO_ZENOH_NAMESPACE on both and UDP 52413 reachable. This script changes no network or firewall setting."
    )
    return 0


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "command", nargs="?", default="check", choices=("check", "sync", "doctor")
    )
    args = ap.parse_args()
    return {"check": cmd_check, "sync": cmd_sync, "doctor": cmd_doctor}[args.command](
        args
    )


if __name__ == "__main__":
    sys.exit(main())
