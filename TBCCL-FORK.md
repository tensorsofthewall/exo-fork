# About this fork

`tensorsofthewall/exo-fork` is a maintained fork of [exo-explore/exo](https://github.com/exo-explore/exo) that carries the integration used by [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl): the `PipelineComm` seam, the `MlxTbccl` instance type and placement, and the runner's byte exchange, which let an exo pipeline move its stage activations over [TBCCL](https://github.com/tensorsofthewall/tbccl) between a Mac (Metal) and a Linux host (CUDA). Everything else is upstream exo. The data path is TCP, including a direct Thunderbolt 4 link; it is not RDMA. The integration is described in [docs/mlx-tbccl.md](docs/mlx-tbccl.md).

## What to install

Use the immutable tag, never `main`:

| Tag | Commit | Contents |
|---|---|---|
| `tbccl-integration-v1` | `1416807b46efd87c3601c69ce5784f51c9dd4c75` | upstream exo `75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6` (2026-10-12) plus 15 integration commits |

exo-tbccl 0.3.0 is qualified against exactly this tag. The tag is never moved; a later integration gets a new tag (`tbccl-integration-v2`, ...) and a new exo-tbccl release. The current pairing is on the [TBCCL ecosystem page](https://tbccl.tensorsofthewall.com/en/latest/project/ecosystem.html).

## Provenance

- `main` of this fork tracks upstream `main` plus the integration. It is a development line, not a supported install target, and is updated by merging upstream and re-validating the integration, never automatically.
- `tools/check_upstream_drift.py` and the weekly `upstream-drift` workflow report how far `main` is behind upstream and which upstream changes touch the files the integration changed (derived from the diff between the upstream base and the integration tag). They only report; they merge nothing and move no tag.
- To see exactly what the integration changes: `git diff 75d8ad2bc1eac25aaf7adb39c0784ad0def8c1b6 tbccl-integration-v1`.

Report problems with the integration in [exo-tbccl](https://github.com/tensorsofthewall/exo-tbccl/issues), not to upstream exo.
