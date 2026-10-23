# About this fork

`tensorsofthewall/exo-fork` is a maintained fork of [exo-explore/exo](https://github.com/exo-explore/exo) that carries the integration used by [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl): the `PipelineComm` seam, the `MlxTbccl` instance type and placement, and the runner's byte exchange, which let an exo pipeline move its stage activations over [TBCCL](https://github.com/tensorsofthewall/tbccl) between a Mac (Metal) and a Linux host (CUDA). Everything else is upstream exo. The data path is TCP, including a direct Thunderbolt 4 link; it is not RDMA. The integration is described in [docs/mlx-tbccl.md](docs/mlx-tbccl.md).

## What to install

Use the immutable tag, never `main`:

| Tag | Commit | Contents |
|---|---|---|
| `tbccl-integration-v3` (current) | recorded in the [exo-tbccl compatibility manifest](https://github.com/tensorsofthewall/exo-tbccl/blob/main/compatibility.json) | `tbccl-integration-v2` plus the setup helper and documentation of this line, and a security update of the trust-store dependencies (see [Security](#security-emudhra-ca-removal-october-2026)); the integration code is unchanged |
| `tbccl-integration-v2` (superseded) | `d1497b28b89618172d273e70e9ac7992811635f5` | upstream exo `bc412e2230f395e652c23426a5eb88dfa515769b` (2026-10-13) plus the unchanged integration and the optional `tbccl` extra; its `uv.lock` pins certifi 2026.1.4 |
| `tbccl-integration-v1` (legacy) | `1416807b46efd87c3601c69ce5784f51c9dd4c75` | upstream exo `75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6` (2026-10-12) plus 15 integration commits; no extra |

exo-tbccl 0.3.0 is the adapter version of v1 and v2 (the wheel did not change), and the `tbccl` extra of v2 and v3 installs that wheel. Tags are never moved; a later integration gets a new tag. The current pairing is on the [TBCCL ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html).

## Installing the TBCCL-compatible fork

Why the fork: upstream exo has no way to run a pipeline stage over TBCCL. The fork adds the `PipelineComm` seam, the `MlxTbccl` instance type and its placement, and the runner byte exchange; the transport itself is the separate [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl) package. Installing the fork does not change exo's default behaviour; without exo-tbccl, selecting `MlxTbccl` is a clear placement error.

**Supported: the tag `tbccl-integration-v3` with the optional `tbccl` extra**, installed on **every** node. Python 3.13, uv, node and Rust are needed as for any exo source install ([README](README.md#run-from-source-macos)). The extra adds convenient installation through uv: `exo-tbccl==0.3.0` (platform-gated release-wheel sources for Linux x86-64 and macOS arm64, recorded with hashes in `uv.lock`). The adapter remains optional (exo works without it), the wheel contains the TBCCL runtime so no separate libtbccl is needed, and on any other platform the extra installs nothing. Use the commands below (the README of v2 and v1 predates the method).

Linux x86-64 (CUDA 13):

```bash
git clone --depth 1 --branch tbccl-integration-v3 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
uv sync --no-dev --locked --extra mlx-cuda13 --extra tbccl
uv run --no-sync python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"
```

macOS arm64 (Metal):

```bash
git clone --depth 1 --branch tbccl-integration-v3 https://github.com/tensorsofthewall/exo-fork.git
cd exo-fork && (cd dashboard && npm install && npm run build)
uv sync --no-dev --locked --extra mlx --extra tbccl
uv run --no-sync python -c "import exo_tbccl; print(exo_tbccl.__version__, exo_tbccl.is_available())"
```

Expected output: `0.3.0 (True, '')`. A repeated sync keeps the adapter and leaves the MLX and PyTorch pins unchanged (verified with `--locked` on Linux x86-64 and macOS arm64). Then follow [docs/mlx-tbccl.md](docs/mlx-tbccl.md) and the [exo-tbccl guide](https://exo-tbccl.tensorsofthewall.com/en/latest/getting-started/install.html) to place an instance.

On macOS the first `uv sync` builds MLX from its pinned git revision (needs Xcode and the Metal toolchain; several minutes), and the dashboard build needs node (`brew install node`), which `uv sync` itself does not. If the adapter is missing or the platform has none (wheels exist only for Linux x86-64 and macOS arm64), exo reports `exo_tbccl is not installed (...)`, and selecting `MlxTbccl` fails placement with `MlxTbccl requires the optional exo-tbccl package: <reason>`.

**Without the dashboard (HTTP API only).** `exo --help` and `--version` work without a built dashboard, but serving needs a dashboard directory. If you only need the API, you do not have to install Node: point `EXO_DASHBOARD_DIR` at any existing directory (relative to `$HOME`, or absolute) and start exo; the API answers on port 52415 and `/` returns 404. Verified on Linux x86-64 with the v2 code and an empty directory (the integration code is the same in v3). Build the dashboard (`cd dashboard && npm install && npm run build`) when you want the web UI.

**Checking a checkout.** On the development line (`main`, not inside the tagged v2 tree) `python3 tools/setup_tbccl.py check` reports, without changing anything, whether uv, git, Rust (and on macOS Xcode and the Metal toolchain) are present, which revision this is, whether the dashboard is built, and prints the exact `uv sync` command for the platform; `sync` runs that locked sync, and `doctor` prints the installed exo, adapter and TBCCL runtime versions, the MLX device, model directories, discovery namespace and whether the ports 52413-52415 are free. It never installs system packages or touches network settings. It is part of `tbccl-integration-v3`; the older tags do not contain it.

**Qualification.** One two-rank `MlxTbccl` pipeline between an Apple Silicon M4 (Metal) host and an RTX 3070 Ti (CUDA) host over a Thunderbolt 4 IP link, `Qwen3-0.6B-8bit`, completed a short generation with no new AER errors or link instability during that session. It does not establish performance, scalability, recovery or other models.

## Security: emSign CA removal (October 2026)

In October 2026 private keys of eight eMudhra intermediate certificates were reported compromised and the emSign root was removed from the CA bundles that Python tooling uses. What it means for this fork:

- exo downloads model files over HTTPS with a context built from the `certifi` package (`src/exo/download/download_utils.py`). `tbccl-integration-v2` and `tbccl-integration-v1` lock certifi 2026.1.4 in `uv.lock`, which still trusts the emSign roots. `tbccl-integration-v3` locks certifi 2026.10.19. Only certifi was changed in the Python lock (no other package moved).
- The `exo_rs` extension compiles the Mozilla root list through `webpki-roots` and `webpki-root-certs` (reached through zenoh's TLS link and rustls-platform-verifier). v3 locks 1.0.11, the newest release, which removes three of the four emSign roots; the remaining emSign Root CA G1 entry follows the upstream list. exo's default networking does not use a TLS link.
- `[tool.uv] required-version` is now `>=0.13.4` and the CI installs uv 0.13.5. uv 0.13.5 installs managed Python builds that carry pip 26.2.2.
- To update an existing v2 or v1 checkout without switching tags, upgrade certifi in exo's environment (`uv pip install --python .venv/bin/python certifi==2026.10.19`); a later exact `uv sync` of the old tag reinstalls the old certifi. Upgrade uv itself to 0.13.4 or later and pip to 26.2.2 or later in any environment you use to install it.
- The tags `tbccl-integration-v1` and `-v2` are immutable and unchanged. This note does not say that either tag is unusable on a trusted network; it is a trust-store update.

## Legacy: `tbccl-integration-v1` and a manual wheel

The earlier immutable tag `tbccl-integration-v1` has no extra. It stays available and unchanged: the exo-tbccl 0.3.0 wheel is installed by hand into the tag's own uv environment on **every** node. Python 3.13, uv, node and Rust are needed as for any exo source install ([README](README.md#run-from-source-macos)).

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

## Provenance

- `main` of this fork tracks upstream `main` plus the integration. It is a development line, not a supported install target, and is updated by merging upstream and re-validating the integration, never automatically.
- `tools/check_upstream_drift.py` and the weekly `upstream-drift` workflow report how far `main` is behind upstream and which upstream changes touch the files the integration changed (derived from the diff between the upstream base and the integration tag). They only report; they merge nothing and move no tag.
- To see exactly what the integration changes: `git diff bc412e2230f395e652c23426a5eb88dfa515769b tbccl-integration-v2` (v2) or `git diff 75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6 tbccl-integration-v1` (v1). `git diff tbccl-integration-v2 tbccl-integration-v3` shows the setup helper, the documentation and the dependency update that v3 adds.

Report problems with the integration in [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl/issues), not to upstream exo.
