# About this fork

`tensorsofthewall/exo-fork` is a maintained fork of [exo-explore/exo](https://github.com/exo-explore/exo) that carries the integration used by [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl): the `PipelineComm` seam, the `MlxTbccl` instance type and placement, and the runner's byte exchange, which let an exo pipeline move its stage activations over [TBCCL](https://github.com/tensorsofthewall/tbccl) between a Mac (Metal) and a Linux host (CUDA). Everything else is upstream exo. The data path is TCP, including a direct Thunderbolt 4 link; it is not RDMA. The integration is described in [docs/mlx-tbccl.md](docs/mlx-tbccl.md).

## What to install

Use the immutable tag, never `main`:

| Tag | Commit | Contents |
|---|---|---|
| `tbccl-integration-v1` | `1416807b46efd87c3601c69ce5784f51c9dd4c75` | upstream exo `75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6` (2026-10-12) plus 15 integration commits |

exo-tbccl 0.3.0 is qualified against exactly this tag. The tag is never moved; a later integration gets a new tag (`tbccl-integration-v2`, ...) and a new exo-tbccl release. The current pairing is on the [TBCCL ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html).

## Installing the TBCCL-compatible fork

Why the fork: upstream exo has no way to run a pipeline stage over TBCCL. The fork adds the `PipelineComm` seam, the `MlxTbccl` instance type and its placement, and the runner byte exchange; the transport itself is the separate [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl) package. Installing the fork does not change exo's default behaviour; without exo-tbccl, selecting `MlxTbccl` is a clear placement error.

**Supported today: the tag `tbccl-integration-v1` plus the exo-tbccl 0.3.0 wheel**, installed into the tag's own uv environment on **every** node. Python 3.13, uv, node and Rust are needed as for any exo source install ([README](README.md#run-from-source-macos)).

Linux x86-64 (CUDA 13):

```bash
git clone --branch tbccl-integration-v1 --depth 1 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
uv sync --no-dev --extra mlx-cuda13
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0/exo_tbccl-0.3.0-cp313-cp313-manylinux_2_24_x86_64.manylinux_2_28_x86_64.whl
.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"
```

macOS arm64 (Metal):

```bash
git clone --branch tbccl-integration-v1 --depth 1 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
uv sync --no-dev --extra mlx
uv pip install --python .venv/bin/python --no-deps https://github.com/tensorsofthewall/exo-tbccl/releases/download/v0.3.0/exo_tbccl-0.3.0-cp313-cp313-macosx_14_0_arm64.whl
.venv/bin/python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"
```

Expected output: `0.3.0 (True, '')`. `--no-deps` keeps uv from touching exo's MLX and PyTorch pins (the wheel has no dependencies anyway). `uv sync` is an exact sync and removes the adapter; after the wheel is installed use `uv sync --inexact ...` (verified to keep it), or reinstall the wheel. Then follow [docs/mlx-tbccl.md](docs/mlx-tbccl.md) and the [exo-tbccl guide](https://exo-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) to place an instance.

**Upstream versus fork:** the upstream instructions in the README install upstream exo (or this fork's `main`, which is a development line) and give no TBCCL pipeline. Use the tag above for TBCCL.

**Preferred workflow once the next integration is released (not yet available).** The development branch declares an optional `tbccl` extra (`exo-tbccl==0.3.0`, with wheel sources for Linux x86-64 and macOS arm64 and no effect on other platforms), so that `uv sync --no-dev --extra mlx-cuda13 --extra tbccl` (Linux) or `uv sync --no-dev --extra mlx --extra tbccl` (macOS) installs the adapter together with exo and a repeated sync keeps it. It is exo-only: the adapter is not a dependency of exo. This is available only from a revision that carries the extra; the supported tag `tbccl-integration-v1` does not, and `tbccl-integration-v1` is never changed. A new integration tag will be created after the extra is qualified on both platforms.

## Provenance

- `main` of this fork tracks upstream `main` plus the integration. It is a development line, not a supported install target, and is updated by merging upstream and re-validating the integration, never automatically.
- `tools/check_upstream_drift.py` and the weekly `upstream-drift` workflow report how far `main` is behind upstream and which upstream changes touch the files the integration changed (derived from the diff between the upstream base and the integration tag). They only report; they merge nothing and move no tag.
- To see exactly what the integration changes: `git diff 75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6 tbccl-integration-v1`.

Report problems with the integration in [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl/issues), not to upstream exo.
