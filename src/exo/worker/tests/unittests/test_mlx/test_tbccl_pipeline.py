"""Synthetic pipeline: unsharded vs MlxRing vs MlxTbccl through exo's real pipeline code (see tbccl_pipeline_driver.py).

Each scenario runs in a fresh interpreter (ranks are spawned: MLX/CUDA must not be forked from a process that already imported it).
Skipped when the optional exo-tbccl package is not installed.
"""

import importlib.util
import json
import subprocess
import sys
from typing import Any, cast

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        importlib.util.find_spec("exo_tbccl") is None,
        reason="optional exo-tbccl package is not installed",
    ),
]

CHUNKS = 3
DRIVER = "exo.worker.tests.unittests.test_mlx.tbccl_pipeline_driver"


def run_driver(backend: str, world: int, dtype_name: str) -> list[dict[str, Any]]:
    proc = subprocess.run(
        [sys.executable, "-m", DRIVER, backend, str(world), dtype_name],
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT "))
    return cast(list[dict[str, Any]], json.loads(line[len("RESULT ") :]))


def check_boundaries(results: list[dict[str, Any]]) -> None:
    for r in range(len(results) - 1):
        sent = cast(list[str], results[r]["sent"])
        recv = cast(list[str], results[r + 1]["recv"])
        # one activation per decode chunk and per prefill chunk crosses each boundary
        assert len(sent) == 2 * CHUNKS and sent == recv, (r, sent, recv)


@pytest.mark.parametrize("dtype_name", ["float32", "bfloat16"])
@pytest.mark.parametrize("world", [2, 3, 4])
def test_tbccl_pipeline_matches_unsharded_and_boundaries_are_bit_exact(
    world: int, dtype_name: str
) -> None:
    results = run_driver("tbccl", world, dtype_name)
    assert all(r["decode_matches_unsharded"] for r in results)
    assert results[-1]["prefill_matches_unsharded"]
    check_boundaries(results)


@pytest.mark.parametrize("world", [2, 3])
def test_ring_and_tbccl_pipelines_move_identical_activation_bytes(world: int) -> None:
    ring = run_driver("ring", world, "bfloat16")
    tbccl = run_driver("tbccl", world, "bfloat16")
    for r in range(world):
        assert ring[r]["decode_matches_unsharded"]
        assert ring[r]["sent"] == tbccl[r]["sent"]
        assert ring[r]["recv"] == tbccl[r]["recv"]
    check_boundaries(ring)
